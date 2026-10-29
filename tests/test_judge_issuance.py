"""tests/test_judge_issuance.py — 판 4 성적서 서버 발급 검사(app.issuance.judge_gate), 심판 라운드 J5b(계획 14-6·14-9a·사-6).

한 DB(임시 파일 SQLite)를 발급 API(get_db)와 서술 작업(jobs.get_session_factory) 두 의존성에 같이 꽂는다 — 따로 꽂으면 발급 쪽이
작업을 못 찾아 "작업 없음"으로 저절로 거절되어 시험에 이빨이 없다. 같은 장치로 "통과 → 발급" 양성 대조를 함께 본다.
작업은 실제 경로(POST /api/narrative/jobs)로 만든다 — 심판은 conftest judge_passes(통과) 또는 키 없음(보류).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, configure_sqlite, get_db
from app.issuance import judge_gate
from app.issuance.bootstrap import DEFAULT_ORGANIZATION
from app.issuance.models import Organization
from app.main import app
from app.narrative import jobs
from app.narrative.models import NarrativeJob, _utcnow

DEV = Path(__file__).parent / "data" / "judge_dev"


@pytest.fixture
def shared(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'shared.db'}", connect_args={"check_same_thread": False})
    configure_sqlite(engine)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        db.add(Organization(**DEFAULT_ORGANIZATION))
        db.commit()

    def db_dep():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = db_dep
    app.dependency_overrides[jobs.get_session_factory] = lambda: factory
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(jobs.get_session_factory, None)
        engine.dispose()


def _ed4() -> dict:
    return json.loads((DEV / "dev_binary.request.json").read_text(encoding="utf-8"))


def _job(client, raw) -> str:
    created = client.post("/api/narrative/jobs", json=raw)
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    assert client.get(f"/api/narrative/jobs/{job_id}").json()["status"] == "done"
    return job_id


def _issue(client, *, run_id="run-1", content=None, job_id=None):
    body = {"run_id": run_id, "model_name": "m", "content": content}
    if job_id is not None:
        body["narrative_job_id"] = job_id
    return client.post("/api/reports/issue", json=body)


def _code(resp) -> str:
    assert resp.status_code == 409, resp.text
    return resp.json()["detail"].rsplit("(", 1)[1].rstrip(")")


ED4 = {"contentSchema": 4, "title": "판 4 성적서"}


def _content(client, job_id: str) -> dict:
    """작업 응답을 프론트가 성적서 데이터로 옮기는 대로(narrativeJobApi.mapNarrativeResult → useReportData.mergeNarrative) 손으로 옮긴
    판 4 content. app.narrative.digest 를 쓰지 않는다(같은 코드로 만들면 대조에 이빨이 없다). 서술 밖 칸은 지문과 상관없음을 보이려고 둔다."""
    n = client.get(f"/api/narrative/jobs/{job_id}").json()["result"]
    return {
        **ED4,
        "meta": {"reportId": ""},
        "interpretation": {
            "confusionAnalysis": n["interpretation"]["confusion_analysis"],
            "distributionAnalysis": n["interpretation"]["distribution_analysis"],
            "additionalObservations": n["interpretation"].get("additional_observations") or "",
        },
        "conclusion": {"verdict": "FAIL", "score": 50, "benchmarkRefs": [],
                       **{k: n["conclusion"][k] for k in ("benchmark", "narrative", "risks")}},
        "recommendationNarrative": {"dataQuality": n["recommendation_narrative"]["data_quality"],
                                    "modelOps": n["recommendation_narrative"]["model_ops"]},
        "recommendations": [
            {"priority": r["priority"].upper(), "category": r["category"], "action": r["action"],
             "expectedImpact": r["expected_impact"],
             **({"problemId": r["problem_id"].strip()} if (r.get("problem_id") or "").strip() else {})}
            for r in n["recommendations"]
        ],
        "narrativeSections": n["section_sources"],
        "judge": n.get("judge"),
    }


# ── 판 판정(프론트 규칙과 같게, 닫힌 쪽) ──


@pytest.mark.parametrize("content, checked", [
    (None, False), ({"smoke": True}, False), ({"contentSchema": None}, False),
    ({"contentSchema": 1}, False), ({"contentSchema": 2}, False), ({"contentSchema": 3}, False),
    ({"contentSchema": 4}, True), ({"contentSchema": "4"}, True), ({"contentSchema": 4.0}, True),
    ({"contentSchema": 3.0}, True), ({"contentSchema": "3"}, True), ({"contentSchema": 5}, True),
    ({"contentSchema": 0}, True), ({"contentSchema": True}, True), ({"contentSchema": [4]}, True),
])
def test_edition_rule_is_closed(content, checked):
    assert judge_gate.content_is_edition4(content) is checked


# ── 첫 발급 ──


def test_smoke_and_edition3_content_are_not_checked(shared):
    client, _ = shared
    assert _issue(client, run_id="ci-smoke", content={"smoke": True}).status_code == 200  # CI 부팅 스모크 그대로
    assert _issue(client, run_id="ed3", content={"contentSchema": 3}).status_code == 200
    assert _issue(client, run_id="meta").status_code == 200


def test_passed_edition4_job_issues_and_every_other_case_is_rejected(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    assert judge_passes  # 심판 자리가 실제로 불렸다(양성 대조가 진짜 통과 기록을 본다)
    ed3_raw = {**_ed4(), "report_edition": 3}
    ed3_raw.pop("judge_material")
    ed3_job = _job(client, ed3_raw)

    assert _code(_issue(client, run_id="a", content=ED4)) == "judge_job_missing"
    assert _code(_issue(client, run_id="b", content=ED4, job_id="  ")) == "judge_job_missing"
    assert _code(_issue(client, run_id="c", content=ED4, job_id="no-such-job")) == "judge_job_not_found"
    assert _code(_issue(client, run_id="d", content=ED4, job_id=ed3_job)) == "judge_job_not_edition4"
    assert _code(_issue(client, run_id="e0", content=ED4, job_id=passed)) == "judge_narrative_mismatch"  # 서술 없는 content
    ok = _issue(client, run_id="e", content=_content(client, passed), job_id=passed)
    assert ok.status_code == 200 and ok.json()["report_no"]


def test_held_job_is_rejected(shared):
    client, _ = shared
    app.state.openai_client = None  # 심판 키 없음 → 보류(no_key)
    held = _job(client, _ed4())
    assert client.get(f"/api/narrative/jobs/{held}").json()["result"]["judge"]["gate"] == "held"
    assert _code(_issue(client, content=ED4, job_id=held)) == "judge_not_passed"


def test_running_and_failed_jobs_are_rejected(shared):
    client, factory = shared
    jobs._db_insert(factory, "job-run", [], {"report_edition": 4}, _utcnow())
    assert _code(_issue(client, run_id="r", content=ED4, job_id="job-run")) == "judge_not_passed"
    jobs._db_finish(factory, "job-run", status="failed", narrative=None,
                    internal={"judge": {"gate": "passed"}}, now=_utcnow())  # 실패인데 기록만 passed — 상태가 먼저
    assert _code(_issue(client, run_id="f", content=ED4, job_id="job-run")) == "judge_not_passed"


def test_expired_job_record_is_rejected(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    with factory() as db:
        job = db.get(NarrativeJob, passed)
        job.created_at = _utcnow() - jobs.record_retention() - timedelta(minutes=1)
        db.commit()
    assert _code(_issue(client, content=ED4, job_id=passed)) == "judge_record_expired"


def test_result_text_expiry_does_not_lose_the_gate(shared, judge_passes, monkeypatch):
    """결과 글은 24시간 뒤 지워져도 gate 는 내부 기록에 남는다(계획 사-4) — 발급은 된다."""
    client, factory = shared
    passed = _job(client, _ed4())
    content = _content(client, passed)  # 프론트는 서술을 run 에 저장해 둔다 — 결과 글이 지워져도 content 에 있다
    with factory() as db:
        db.get(NarrativeJob, passed).result_json = None
        db.commit()
    assert _issue(client, content=content, job_id=passed).status_code == 200


# ── 정정 발급(14-9a) ──


def test_reissue_has_the_same_check(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    good = _content(client, passed)
    issued = _issue(client, run_id="run-r", content=good, job_id=passed).json()
    no = issued["report_no"]

    def reissue(job_id=None, content=good):
        body = {"run_id": "run-r", "note": "정정", "content": content}
        if job_id is not None:
            body["narrative_job_id"] = job_id
        return client.post(f"/api/reports/{no}/reissue", json=body)

    assert _code(reissue()) == "judge_job_missing"
    assert _code(reissue("no-such-job")) == "judge_job_not_found"
    app.state.openai_client = None
    held = _job(client, _ed4())
    assert _code(reissue(held)) == "judge_not_passed"
    assert _code(reissue(passed, _tampered(good, ("conclusion", "narrative")))) == "judge_narrative_mismatch"
    assert _code(reissue(passed, ED4)) == "judge_narrative_mismatch"
    assert reissue(passed).status_code == 200
    assert reissue(None, {"contentSchema": 3}).status_code == 200  # 판 3 내용은 검사하지 않는다


def test_job_id_field_is_optional_and_bounded():
    from app.issuance.schemas import IssueRequest, ReissueRequest

    assert IssueRequest(run_id="x").narrative_job_id is None
    with pytest.raises(Exception):
        IssueRequest(run_id="x", narrative_job_id="j" * 65)
    assert ReissueRequest(run_id="x", note="n").narrative_job_id is None


# ── 서술 글 지문 대조(14-9b, J9) — 같은 공유 DB 장치 ──


def _tampered(content: dict, path: tuple, value="바뀐 글") -> dict:
    """content 를 깊게 복사해 path 의 값만 바꾼다(정수는 목록 자리)."""
    out = json.loads(json.dumps(content))
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is _DROP:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


_DROP = object()

NARRATIVE_PATHS = [
    ("interpretation", "confusionAnalysis"), ("interpretation", "distributionAnalysis"),
    ("interpretation", "additionalObservations"),
    ("conclusion", "benchmark"), ("conclusion", "narrative"), ("conclusion", "risks"),
    ("recommendationNarrative", "dataQuality"), ("recommendationNarrative", "modelOps"),
    ("recommendations", 0, "priority"), ("recommendations", 0, "category"), ("recommendations", 0, "action"),
    ("recommendations", 0, "expectedImpact"), ("recommendations", 0, "problemId"),
]


@pytest.fixture
def passed_job(shared, judge_passes):
    client, factory = shared
    job_id = _job(client, _ed4())
    content = _content(client, job_id)
    assert content["recommendations"] and "problemId" in content["recommendations"][0]  # 권고 칸까지 대조가 닿는 픽스처
    assert content["interpretation"]["additionalObservations"] is not None
    return client, factory, job_id, content


def test_digest_is_recorded_only_for_passed_jobs(shared, judge_passes):
    client, factory = shared
    passed = _job(client, _ed4())
    with factory() as db:
        rec = json.loads(db.get(NarrativeJob, passed).internal_json)["judge"]
    assert isinstance(rec["narrative_sha256"], str) and len(rec["narrative_sha256"]) == 64
    assert "narrative_sha256" not in client.get(f"/api/narrative/jobs/{passed}").json()["result"]["judge"]  # 고객 묶음엔 없음
    app.state.openai_client = None
    held = _job(client, _ed4())
    with factory() as db:
        assert "narrative_sha256" not in json.loads(db.get(NarrativeJob, held).internal_json)["judge"]


@pytest.mark.parametrize("path", NARRATIVE_PATHS, ids=lambda p: ".".join(map(str, p)))
def test_any_narrative_change_is_rejected(passed_job, path):
    client, _, job_id, content = passed_job
    assert _code(_issue(client, run_id="t", content=_tampered(content, path), job_id=job_id)) == "judge_narrative_mismatch"


@pytest.mark.parametrize("path", NARRATIVE_PATHS[:8], ids=lambda p: ".".join(map(str, p)))
def test_missing_narrative_field_is_rejected(passed_job, path):
    client, _, job_id, content = passed_job
    assert _code(_issue(client, run_id="t", content=_tampered(content, path, _DROP), job_id=job_id)) == "judge_narrative_mismatch"


@pytest.mark.parametrize("change", [
    lambda c: _tampered(c, ("conclusion", "narrative"), c["conclusion"]["narrative"] + " "),  # 끝 공백 하나
    lambda c: _tampered(c, ("conclusion", "narrative"), c["conclusion"]["narrative"].replace("다", "다\n", 1)),
    lambda c: _tampered(c, ("recommendations",), c["recommendations"][::-1] if len(c["recommendations"]) > 1
                        else c["recommendations"] * 2),  # 순서·개수
    lambda c: _tampered(c, ("recommendations",), c["recommendations"][:-1]),
    lambda c: _tampered(c, ("recommendations", 0, "priority"), c["recommendations"][0]["priority"].lower()),
    lambda c: _tampered(c, ("recommendations", 0, "problemId"), " " + c["recommendations"][0]["problemId"]),
], ids=["끝공백", "줄바꿈", "권고순서", "권고빠짐", "우선순위소문자", "문제번호공백"])
def test_content_side_is_not_normalized(passed_job, change):
    """content 쪽은 프론트가 저장한 값 그대로 — 공백·줄바꿈·순서·대소문자를 너그럽게 보지 않는다."""
    client, _, job_id, content = passed_job
    assert _code(_issue(client, run_id="t", content=change(content), job_id=job_id)) == "judge_narrative_mismatch"


def test_non_narrative_fields_do_not_matter(passed_job):
    """판정·점수·8.1 표·칸 출처·심판 묶음·제목 같은 서술 밖 칸은 지문에 들지 않는다(그건 코드 값이고 다른 검사 몫)."""
    client, _, job_id, content = passed_job
    other = _tampered(_tampered(_tampered(content, ("conclusion", "verdict"), "PASS"), ("title",), "다른 제목"),
                      ("narrativeSections",), {})
    other.pop("judge")
    assert _issue(client, run_id="t", content=other, job_id=job_id).status_code == 200


def test_other_passed_jobs_narrative_is_rejected(shared, judge_passes):
    """핵심 갈래 — 통과한 다른 작업의 번호를 붙여도 서술이 다르면 거절(J5b 의 남은 구멍)."""
    client, _ = shared
    a = _job(client, _ed4())
    b = _job(client, json.loads((DEV / "dev_multiclass.request.json").read_text(encoding="utf-8")))
    ca, cb = _content(client, a), _content(client, b)
    assert ca["conclusion"]["narrative"] != cb["conclusion"]["narrative"]
    assert _code(_issue(client, run_id="x", content=cb, job_id=a)) == "judge_narrative_mismatch"
    assert _code(_issue(client, run_id="y", content=ca, job_id=b)) == "judge_narrative_mismatch"
    assert _issue(client, run_id="x", content=ca, job_id=a).status_code == 200
    assert _issue(client, run_id="y", content=cb, job_id=b).status_code == 200


def test_passed_record_without_digest_is_rejected(shared):
    client, factory = shared
    jobs._db_insert(factory, "job-old", [], {"report_edition": 4}, _utcnow())
    jobs._db_finish(factory, "job-old", status="done", narrative=None,
                    internal={"report_edition": 4, "judge": {"gate": "passed"}}, now=_utcnow())
    assert _code(_issue(client, content=ED4, job_id="job-old")) == "judge_narrative_unrecorded"
    jobs._db_insert(factory, "job-bad", [], {"report_edition": 4}, _utcnow())
    jobs._db_finish(factory, "job-bad", status="done", narrative=None,
                    internal={"report_edition": 4, "judge": {"gate": "passed", "narrative_sha256": 7}}, now=_utcnow())
    assert _code(_issue(client, run_id="z", content=ED4, job_id="job-bad")) == "judge_narrative_unrecorded"


def test_reissue_rejects_other_jobs_narrative(shared, judge_passes):
    client, _ = shared
    a = _job(client, _ed4())
    b = _job(client, json.loads((DEV / "dev_multiclass.request.json").read_text(encoding="utf-8")))
    no = _issue(client, run_id="run-q", content=_content(client, a), job_id=a).json()["report_no"]

    def reissue(job_id, content):
        return client.post(f"/api/reports/{no}/reissue",
                           json={"run_id": "run-q", "note": "정정", "content": content, "narrative_job_id": job_id})

    assert _code(reissue(a, _content(client, b))) == "judge_narrative_mismatch"
    assert reissue(b, _content(client, b)).status_code == 200  # 다시 만든 서술로 정정 — 그 작업 번호와 그 글이면 받는다


def test_smoke_and_edition3_do_not_touch_the_digest(shared, monkeypatch):
    """판 3 내용·{"smoke": true} 는 지문 대조를 타지 않는다(대조 함수가 불리지 않음)."""
    from app.narrative import digest

    def boom(_c):
        raise AssertionError("판 3·스모크에서 지문을 만들었다")

    monkeypatch.setattr(digest, "of_content", boom)
    client, _ = shared
    assert _issue(client, run_id="s", content={"smoke": True}).status_code == 200
    assert _issue(client, run_id="e3", content={"contentSchema": 3, "interpretation": {}}).status_code == 200


# ── J10 검토 ⑥ 통과 기록의 검증 기록 잠금 값(P1-8 나) — 공유 DB 고정 장치 ──


def _set_lock(factory, job_id, value) -> None:
    with factory() as db:
        job = db.get(NarrativeJob, job_id)
        internal = json.loads(job.internal_json)
        if value is None:
            internal["judge"].pop("criteria_lock", None)
        else:
            internal["judge"]["criteria_lock"] = value
        job.internal_json = json.dumps(internal, ensure_ascii=False)
        db.commit()


@pytest.mark.parametrize("value, lock_env, ok", [
    ("ok", "off", True), ("ok", None, True),          # 검증한 기준으로 채점 — 늘 받음
    ("off", "off", True),                             # 잠금을 끈 개발·시험 서버의 기록 — 지금도 꺼져 있으면 받음
    ("off", None, False), ("off", "on", False),       # 지금 잠금이 켜져 있으면(운영) 꺼진 채 채점한 기록은 거절
    (None, "off", False), ("unvalidated", "off", False), ("error:OSError", "off", False), ("", "off", False),
])
def test_issuance_checks_the_recorded_criteria_lock(shared, judge_passes, monkeypatch, value, lock_env, ok):
    from app.judge import criteria_lock

    client, factory = shared
    job_id = _job(client, _ed4())
    assert json.loads(_row_internal(factory, job_id))["judge"]["criteria_lock"] == "off"  # conftest 가 끈 채 채점
    content = _content(client, job_id)
    _set_lock(factory, job_id, value)
    if lock_env is None:
        monkeypatch.delenv(criteria_lock.ENV, raising=False)
    else:
        monkeypatch.setenv(criteria_lock.ENV, lock_env)
    resp = _issue(client, content=content, job_id=job_id)
    if ok:
        assert resp.status_code == 200, resp.text
    else:
        assert _code(resp) == "judge_criteria_unverified"


def _row_internal(factory, job_id) -> str:
    with factory() as db:
        return db.get(NarrativeJob, job_id).internal_json


# ── P1-7(J10 검토): 고객 글에 내부 낱말 금지 — 사용자 결정 문안(2026-10-29)으로 바꾼 뒤 보통 시험 ──

INTERNAL_WORDS = ("심판", "판 4", "판4", "ai", "llm", "gpt", "claude", "sonnet", "opus", "haiku", "모델")


@pytest.mark.parametrize("code", sorted(judge_gate.MESSAGES))
def test_issuance_messages_have_no_internal_words(code):
    msg = judge_gate.MESSAGES[code]
    low = msg.lower()
    for word in INTERNAL_WORDS:
        assert word not in low, (word, msg)
    assert msg.endswith("주세요.") or msg.endswith("습니다."), msg  # 합니다체 문장으로 끝남


def test_new_reason_code_is_in_the_wording_check():
    assert set(judge_gate.MESSAGES) == {
        "judge_job_missing", "judge_job_not_found", "judge_job_not_edition4", "judge_record_expired",
        "judge_not_passed", "judge_narrative_unrecorded", "judge_narrative_mismatch", "judge_criteria_unverified",
    }


def test_rejection_detail_is_the_message_then_code(passed_job):
    """고객에게 보이는 글 꼴: "{MESSAGES[code]} ({code})" — 프론트 B48 이 이 글을 그대로 싣는다."""
    client, _factory, _job_id, content = passed_job
    resp = _issue(client, content=content, job_id=None)
    assert resp.json()["detail"] == f"{judge_gate.MESSAGES['judge_job_missing']} (judge_job_missing)"


def test_lone_surrogate_in_content_is_rejected_not_500(passed_job):
    """서술 칸에 짝 없는 대리 글자가 있어도 발급 검사는 예외가 아니라 거절 사유(닫힌 쪽). API 앞단(요청 JSON 해석)은 이 글자를
    422 로 막는데, 그 오류 응답을 만들 때 프레임워크가 500 을 낸다 — 발급 검사와 별개라 J10 인계에 따로 적었다."""
    client, factory, job_id, content = passed_job
    bad = {**content, "conclusion": {**content["conclusion"], "narrative": "총평 \ud800"}}
    with factory() as db:
        assert judge_gate.reject_reason(db, bad, job_id) == "judge_narrative_mismatch"
        assert judge_gate.reject_reason(db, content, job_id) is None
