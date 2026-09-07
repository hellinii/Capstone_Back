"""tests/test_reissue_changes.py — 재발급이 무엇을 바꿨는지 남긴다.

ISSUES.md F-06 (2026-09-07 ★결정 4 — 앞으로 발급되는 것만 정정).

재발급 이력에는 버전·발급자·시각·비고만 있었다. **무엇이 바뀌었는지**는 어디에도
없어서, 정정 발급을 받은 사람은 "v1.1 이 나왔다"는 사실만 알고 무엇이 달라졌는지
알 수 없었다.

필요한 데이터는 **이미 전부 서버에 있다** — 차수마다 성적서 원본 스냅샷을 보관하기
때문이다(F-01). 새 컬럼을 만들지 않고 **읽는 시점에 인접 스냅샷을 대조**해 만든다.
`issuance` 테이블에 컬럼을 추가하지 않는 것은 F-11 의 규칙이기도 하다
(`create_all` 은 기존 테이블에 컬럼을 추가하지 못한다).
"""
from app.issuance.serializers import diff_report_sections


def test_changed_top_level_sections_are_listed():
    before = {"kpiResults": [{"metricId": "M1", "value": 0.9}], "conclusion": {"verdict": "PASS"}}
    after = {"kpiResults": [{"metricId": "M1", "value": 0.8}], "conclusion": {"verdict": "PASS"}}

    assert diff_report_sections(before, after) == ["kpiResults"]


def test_unchanged_content_yields_no_changes():
    same = {"kpiResults": [], "conclusion": {"verdict": "PASS"}}

    assert diff_report_sections(same, dict(same)) == []


def test_added_and_removed_sections_are_reported():
    assert diff_report_sections({"a": 1}, {"a": 1, "b": 2}) == ["b"]
    assert diff_report_sections({"a": 1, "b": 2}, {"a": 1}) == ["b"]


def test_multiple_changes_are_sorted_for_determinism():
    before = {"z": 1, "a": 1, "m": 1}
    after = {"z": 2, "a": 2, "m": 1}

    assert diff_report_sections(before, after) == ["a", "z"]


def test_missing_previous_snapshot_means_unknown():
    """이전 차수 스냅샷이 없으면(소급 백필을 하지 않았으므로 구 발급본) 추측하지 않는다."""
    assert diff_report_sections(None, {"a": 1}) is None
    assert diff_report_sections({"a": 1}, None) is None


def test_history_reports_changed_sections_end_to_end(tmp_path, monkeypatch):
    """실제 발급 → 재발급 경로에서 이력에 변경 절이 실린다."""
    import json

    from app.issuance.serializers import issuance_out

    class _Snapshot:
        def __init__(self, version, content):
            self.version = version
            self.content_json = json.dumps(content)

    class _Issuance:
        def __init__(self, version, note):
            self.version = version
            self.note = note
            self.issuer = "발급부"
            from datetime import datetime, timezone
            self.issued_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class _Report:
        report_no = "RPT-2026-0001"
        current_version = "v1.1"
        issuances = [_Issuance("v1.0", "최초 발급"), _Issuance("v1.1", "정정 발급")]
        snapshots = [
            _Snapshot("v1.0", {"kpiResults": [1], "conclusion": {"verdict": "PASS"}}),
            _Snapshot("v1.1", {"kpiResults": [2], "conclusion": {"verdict": "PASS"}}),
        ]
        organization = None

    monkeypatch.setattr(
        "app.issuance.serializers._organization_at_issue",
        lambda report: __import__("app.issuance.schemas", fromlist=["OrganizationOut"]).OrganizationOut(
            org_name="x", evaluator="y", contact="z",
        ),
    )

    out = issuance_out(_Report())

    assert out.history[0].changed_sections is None, "최초 발급은 비교 대상이 없다"
    assert out.history[1].changed_sections == ["kpiResults"]
