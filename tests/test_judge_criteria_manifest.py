"""tests/test_judge_criteria_manifest.py — 심판 기준 묶음의 지문(설계 7.4·7.5, 계획 17-1).

- 파일 ↔ MANIFEST: 기준 파일이 MANIFEST 를 다시 쓰지 않고 바뀌면 빨개진다.
- 버전 기록: VERSION 이 고정 버전(이름에 -dev 가 없음)이면 VERSIONS.json 의 그 버전 지문 == 지금 지문. 고정 뒤 MANIFEST 를
  다시 써서 초록으로 만들려 해도 기록과 달라 빨개진다 → 버전을 올리고 다시 검증해야 한다.
- 기록은 덧붙이기만: 아래 KNOWN 에 적은 줄이 VERSIONS.json 에 글자 그대로 남아 있어야 한다(지우거나 바꾸면 빨개짐).
  새 버전을 고정할 때 VERSIONS.json 과 KNOWN 에 같은 줄을 더한다.
- CHANGELOG 에 지금 VERSION 절이 있다. model.json 이 코드 기본값(app/judge/config)과 같다.
"""
from __future__ import annotations

import re

from app.judge import config, criteria

# 고정한 버전 → 기준 지문. 덧붙이기만 한다(V1 에서 "1.0" 을 처음 적는다).
KNOWN: dict[str, str] = {}


def test_manifest_matches_files():
    files = criteria.bundle_files()
    assert len(files) >= 3  # VERSION·model.json·로더는 늘 있다
    assert criteria.recorded_manifest() == criteria.manifest_text(), (
        "기준 파일이 바뀌었는데 MANIFEST 가 그대로다 — 이유를 CHANGELOG 에 적고 scripts/judge_criteria_manifest.py 로 다시 쓴다"
        " (고정한 버전이면 VERSION 을 올리고 다시 검증)"
    )


def test_manifest_excludes_records_and_lists_relative_paths():
    names = [line.split("  ", 1)[1] for line in criteria.recorded_manifest().splitlines()]
    assert names == sorted(names)
    assert not set(names) & criteria.NOT_IN_MANIFEST
    assert "VERSION" in names and "model.json" in names


def test_fixed_version_fingerprint_is_recorded():
    assert criteria.version_problem() is None


def test_version_problem_cases():
    fp = "a" * 64
    assert criteria.version_problem("1.0-dev", {}, fp) is None
    assert criteria.version_problem("1.0-dev", {"1.0-dev": fp}, fp) is not None
    assert criteria.version_problem("1.0", {"1.0": fp}, fp) is None
    assert criteria.version_problem("1.0", {}, fp) is not None
    assert "VERSION 을 올리고" in criteria.version_problem("1.0", {"1.0": "b" * 64}, fp)


def test_version_records_are_append_only():
    recorded = criteria.recorded_versions()
    assert {k: recorded.get(k) for k in KNOWN} == KNOWN
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in recorded.values())


def test_changelog_has_current_version():
    assert f"## {criteria.version()}\n" in (criteria.HERE / "CHANGELOG.md").read_text(encoding="utf-8")


def test_model_settings_match_code_defaults():
    m = criteria.load_json("model.json")
    assert (m["judge_model"], m["extract_model"], m["effort"], m["extract_effort"]) == (
        config.DEFAULT_JUDGE_MODEL, config.DEFAULT_EXTRACT_MODEL, config.DEFAULT_EFFORT, config.DEFAULT_EXTRACT_EFFORT
    )


def test_fingerprint_changes_when_a_file_changes(tmp_path, monkeypatch):
    # 이빨: 파일 한 바이트가 바뀌면 MANIFEST 글과 지문이 달라진다.
    for p in criteria.bundle_files():
        dst = tmp_path / p.relative_to(criteria.HERE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    monkeypatch.setattr(criteria, "HERE", tmp_path)
    before = criteria.manifest_text()
    (tmp_path / "model.json").write_text((tmp_path / "model.json").read_text() + " ", encoding="utf-8")
    after = criteria.manifest_text()
    assert before != after and criteria.fingerprint(before) != criteria.fingerprint(after)
