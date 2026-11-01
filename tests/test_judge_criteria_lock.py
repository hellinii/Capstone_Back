"""tests/test_judge_criteria_lock.py — 검증 기록 잠금(app.judge.criteria_lock), 심판 라운드 J9(계획 14-10·정9).

conftest 의 _criteria_lock_off 가 모든 시험에서 잠금을 끄므로, 여기서는 monkeypatch.delenv 로 다시 켠다(기본값 = 켜짐).
작업은 실제 경로(POST /api/narrative/jobs)로 만들고, 같은 DB 를 꽂은 발급 API 로 "잠긴 작업은 발급도 거절"을 함께 본다.
"""
from __future__ import annotations

import json

import pytest

from app.judge import criteria, criteria_lock
from app.narrative.models import NarrativeJob
from tests.test_judge_issuance import ED4, _code, _ed4, _issue, _job, shared  # noqa: F401 — 픽스처


def _record(tmp_path, **fields):
    path = tmp_path / "validated.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _current_fp() -> str:
    return criteria.fingerprint(criteria.manifest_text())


@pytest.fixture
def lock_on(monkeypatch):
    monkeypatch.delenv(criteria_lock.ENV, raising=False)


@pytest.fixture
def fixed_version(monkeypatch):
    """기준을 '1.0'(고정된 버전)으로 본다 — 저장소 VERSION(잠그기 전 1.0-dev, 뒤 1.0)과 상관없이."""
    monkeypatch.setattr(criteria, "version", lambda: "1.0")


# ── 켜고 끄기(닫힌 쪽) ──


@pytest.mark.parametrize("value, on", [
    (None, True), ("", True), ("0", True), ("false", True), ("of", True), ("disable", True),
    ("off", False), (" OFF ", False), ("Off", False),
])
def test_only_off_disables(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(criteria_lock.ENV, raising=False)
    else:
        monkeypatch.setenv(criteria_lock.ENV, value)
    assert criteria_lock.enabled() is on


def test_off_never_locks(tmp_path, monkeypatch):
    monkeypatch.setenv(criteria_lock.ENV, "off")
    assert criteria_lock.problem(tmp_path / "없음.json") is None


# ── 판정 ──


def test_no_record_means_unvalidated(lock_on, tmp_path):
    assert criteria_lock.problem(tmp_path / "없음.json") == criteria_lock.UNVALIDATED


def test_repo_record_matches_the_current_criteria(lock_on):
    """J11(2026-11-01, 민서 결정 — V2 부분 검증 수용): 기록은 지금 기준(버전·지문)과 같아야 한다. 기준을 바꾸고 기록을 고치지 않으면
    이 시험이 깨진다 — '배포되는 기준 = 검증한 기준'(정9)을 지키는 문."""
    assert criteria_lock.RECORD_PATH.exists()
    assert criteria_lock.RECORD_PATH.parent == criteria.HERE.parent  # 기준 묶음 폴더 밖(쓰면 지문이 바뀌지 않게)
    rec = json.loads(criteria_lock.RECORD_PATH.read_text(encoding="utf-8"))
    assert (rec["version"], rec["fingerprint"]) == (criteria.version(), criteria.fingerprint())
    assert rec["tag"] == f"judge-criteria-v{criteria.version()}" and "부분 검증" in rec["note"]
    assert criteria_lock.problem() is None


@pytest.mark.parametrize("text", ["{", "[]", '"1.0"', '{"version": "1.0"}', '{"fingerprint": "x"}',
                                  '{"version": 1.0, "fingerprint": "x"}'])
def test_unreadable_record(lock_on, fixed_version, tmp_path, text):
    path = tmp_path / "validated.json"
    path.write_text(text, encoding="utf-8")
    assert criteria_lock.problem(path) == criteria_lock.UNREADABLE


def test_matching_record_unlocks(lock_on, fixed_version, tmp_path):
    assert criteria_lock.problem(_record(tmp_path, version="1.0", fingerprint=_current_fp())) is None


def test_dev_version_is_locked_even_with_matching_record(lock_on, tmp_path, monkeypatch):
    # 저장소 VERSION 에 기대지 않고 개발 중 버전을 만든다(잠그기 전·뒤 모두 같은 것을 본다).
    monkeypatch.setattr(criteria, "version", lambda: "1.0-dev")
    path = _record(tmp_path, version=criteria.version(), fingerprint=_current_fp())
    assert criteria_lock.problem(path) == criteria_lock.DEV_VERSION


def test_version_mismatch(lock_on, fixed_version, tmp_path):
    path = _record(tmp_path, version="1.1", fingerprint=_current_fp())
    assert criteria_lock.problem(path) == criteria_lock.VERSION_MISMATCH


def test_fingerprint_mismatch(lock_on, fixed_version, tmp_path):
    path = _record(tmp_path, version="1.0", fingerprint="0" * 64)
    assert criteria_lock.problem(path) == criteria_lock.FINGERPRINT_MISMATCH


def test_fingerprint_is_from_files_not_from_written_manifest(lock_on, fixed_version, tmp_path, monkeypatch):
    """적어 둔 MANIFEST 와 기록이 같아도 파일이 바뀌었으면 잠긴다(서버에서 기준 파일이 몰래 바뀐 경우)."""
    path = _record(tmp_path, version="1.0", fingerprint=criteria.fingerprint())
    real = criteria.manifest_text()
    monkeypatch.setattr(criteria, "manifest_text", lambda: real.replace("checklist.json", "checklist2.json"))
    assert criteria_lock.problem(path) == criteria_lock.FINGERPRINT_MISMATCH


# ── 작업 흐름 ──


def _internal(factory, job_id) -> dict:
    with factory() as db:
        return json.loads(db.get(NarrativeJob, job_id).internal_json)


def test_locked_job_is_held_without_calling_the_judge(shared, judge_passes, lock_on, tmp_path, monkeypatch):
    monkeypatch.setattr(criteria_lock, "RECORD_PATH", tmp_path / "no_record.json")  # 기록이 없는 서버(검증 전)
    client, factory = shared
    job_id = _job(client, _ed4())
    judge = client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]
    assert (judge["gate"], judge["held_kind"], judge["retry_allowed"], judge["counts"]) == ("held", "judge_error", False, None)
    assert judge_passes == []  # 심판을 부르지 않았다(돈을 쓰지 않음)
    rec = _internal(factory, job_id)["judge"]
    assert (rec["gate"], rec["held_kind"], rec["failure"], rec["criteria_lock"]) == \
        ("held", "judge_error", "criteria_lock", criteria_lock.UNVALIDATED)
    assert _code(_issue(client, content=ED4, job_id=job_id)) == "judge_not_passed"  # 발급도 막힌다


def test_validated_criteria_pass(shared, judge_passes, lock_on, fixed_version, tmp_path, monkeypatch):
    """양성 대조 — 같은 장치에서 기록이 맞으면 통과(잠금 시험이 늘 보류라서 초록인 것이 아님)."""
    monkeypatch.setattr(criteria_lock, "RECORD_PATH", _record(tmp_path, version="1.0", fingerprint=_current_fp()))
    client, factory = shared
    job_id = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]["gate"] == "passed"
    assert len(judge_passes) == 1
    assert _internal(factory, job_id)["judge"]["criteria_lock"] == "ok"


def test_lock_off_is_recorded(shared, judge_passes):
    client, factory = shared
    job_id = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]["gate"] == "passed"
    assert _internal(factory, job_id)["judge"]["criteria_lock"] == "off"


def test_lock_check_error_is_held(shared, judge_passes, lock_on, monkeypatch):
    """잠금 판정 자체가 터져도(기준 파일을 못 읽음 등) 통과로 새지 않는다."""
    def boom():
        raise PermissionError("기준 파일을 못 읽음")

    monkeypatch.setattr(criteria_lock, "problem", boom)
    client, factory = shared
    job_id = _job(client, _ed4())
    rec = _internal(factory, job_id)["judge"]
    assert (rec["gate"], rec["held_kind"], rec["criteria_lock"]) == ("held", "judge_error", "error:PermissionError")
    assert judge_passes == []


def test_edition3_is_not_touched_by_the_lock(shared, judge_passes, lock_on):
    client, factory = shared
    raw = {**_ed4(), "report_edition": 3}
    raw.pop("judge_material")
    job_id = _job(client, raw)
    assert "judge" not in client.get(f"/api/narrative/jobs/{job_id}").json()["result"]
    assert "judge" not in _internal(factory, job_id) and judge_passes == []


def test_no_key_held_comes_before_the_lock(shared, lock_on):
    """심판 키 없음은 만들 때 정해진 보류 — 잠금보다 먼저(CI 부팅 블록의 no_key 단정이 그대로)."""
    from app.main import app

    app.state.openai_client = None
    client, factory = shared
    job_id = _job(client, _ed4())
    judge = client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]
    assert (judge["gate"], judge["held_kind"]) == ("held", "no_key")
    assert "criteria_lock" not in _internal(factory, job_id)["judge"]


def test_dev_server_turns_the_lock_off():
    text = (criteria.HERE.parents[2] / "scripts" / "dev_fake_llm_server.py").read_text(encoding="utf-8")
    assert 'os.environ["JUDGE_CRITERIA_LOCK"] = "off"' in text
    main = (criteria.HERE.parents[2] / "app" / "main.py").read_text(encoding="utf-8")
    assert "JUDGE_CRITERIA_LOCK" not in main  # 운영 진입점은 끄지 않는다


# ── J10 검토: 환경변수로 바꾼 심판 모델·추론 강도(P1-10) ──


@pytest.mark.parametrize("env, value, field", [
    ("JUDGE_MODEL", "gpt-6.1-sol-mini", "judge_model"), ("JUDGE_EXTRACT_MODEL", "gpt-6.1-sol", "extract_model"),
    ("JUDGE_EFFORT", "high", "effort"), ("JUDGE_EXTRACT_EFFORT", "medium", "extract_effort"),
])
def test_env_model_or_effort_change_locks(lock_on, fixed_version, tmp_path, monkeypatch, env, value, field):
    path = _record(tmp_path, version="1.0", fingerprint=_current_fp())
    assert criteria_lock.problem(path) is None and criteria_lock.model_drift() == {}  # 양성 대조
    monkeypatch.setenv(env, value)
    assert criteria_lock.problem(path) == criteria_lock.MODEL_MISMATCH
    drift = criteria_lock.model_drift()
    assert list(drift) == [field] and drift[field]["actual"] == value
    assert drift[field]["criteria"] == criteria.load_json("model.json")[field]


def test_env_same_as_criteria_does_not_lock(lock_on, fixed_version, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", criteria.load_json("model.json")["judge_model"])
    monkeypatch.setenv("JUDGE_EFFORT", "  MEDIUM ")  # config 가 실제로 쓸 값(medium)으로 본다
    assert criteria_lock.problem(_record(tmp_path, version="1.0", fingerprint=_current_fp())) is None


def test_model_drift_job_is_held_and_records_what_differed(shared, judge_passes, lock_on, fixed_version, tmp_path, monkeypatch):
    monkeypatch.setattr(criteria_lock, "RECORD_PATH", _record(tmp_path, version="1.0", fingerprint=_current_fp()))
    monkeypatch.setenv("JUDGE_MODEL", "검증안한-모델")
    client, factory = shared
    job_id = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]["held_kind"] == "judge_error"
    rec = _internal(factory, job_id)["judge"]
    assert rec["criteria_lock"] == criteria_lock.MODEL_MISMATCH and judge_passes == []
    assert rec["model_drift"] == {"judge_model": {"criteria": "gpt-6.1-sol", "actual": "검증안한-모델"}}


def test_model_drift_with_lock_off_is_only_recorded(shared, judge_passes, monkeypatch):
    monkeypatch.setenv("JUDGE_EFFORT", "high")
    client, factory = shared
    job_id = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["result"]["judge"]["gate"] == "passed"
    rec = _internal(factory, job_id)["judge"]
    assert rec["criteria_lock"] == "off" and rec["model_drift"]["effort"] == {"criteria": "medium", "actual": "high"}
