"""tests/test_golden_utils.py — 골든 헬퍼 자체의 진단 품질 검증.

ISSUES.md H-02 — 골든 스냅샷은 오라클이 아니라 현행 출력의 복사본이다. 그 성질 자체는
골든을 고쳐서 해결되지 않지만(H-01 의 기대값 스위트가 옆에 있어야 한다), **불일치가
났을 때 무엇이 달라졌는지 읽을 수 없다**는 것은 별개의 결함이고 즉시 고칠 수 있다.

실측 근거: A-08(multilabel 평균 방식)을 실제로 패치해 전체 스위트를 돌리면
186 passed / 1 failed 인데, 그 유일한 실패 메시지가 다음과 같았다.

    expected keys=['class_distribution', 'dropped_rows', 'results', 'warnings']
      actual keys=['class_distribution', 'dropped_rows', 'results', 'warnings']

양쪽이 글자 그대로 같다. 골든이 흔들리는 모든 변경(A-08 / D-01 / D-06 / C-03)에서
회귀 신호가 판독 불가능하다는 뜻이다.
"""
import json

import pytest

from tests import golden_utils


@pytest.fixture
def golden_dir(tmp_path, monkeypatch):
    """골든 디렉터리를 임시 경로로 돌려 실제 tests/golden/ 을 건드리지 않는다."""
    monkeypatch.setattr(golden_utils, "GOLDEN_DIR", tmp_path)
    monkeypatch.delenv("UPDATE_GOLDEN", raising=False)
    return tmp_path


def _seed(golden_dir, name, payload):
    (golden_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_mismatch_message_names_the_differing_path(golden_dir):
    """[H-02] 깊은 곳의 값 하나가 달라지면 그 경로와 두 값이 메시지에 나와야 한다.

    A-08 회귀와 같은 형태 — 최상위 키 집합은 동일하고 results 안의 한 지표만 바뀐다.
    """
    _seed(golden_dir, "sample", {"results": {"M2": 0.4906396261785457}, "dropped_rows": 1})

    with pytest.raises(AssertionError) as exc:
        golden_utils.assert_golden("sample", {"results": {"M2": 0.495}, "dropped_rows": 1})

    msg = str(exc.value)
    assert "results.M2" in msg, f"달라진 경로가 메시지에 없다:\n{msg}"
    assert "0.49064" in msg, f"기대값이 메시지에 없다:\n{msg}"
    assert "0.495" in msg, f"실제값이 메시지에 없다:\n{msg}"


def test_mismatch_message_reports_added_and_removed_keys(golden_dir):
    """[H-02] 키가 사라지거나 새로 생긴 것도 경로로 지목되어야 한다."""
    _seed(golden_dir, "sample", {"results": {"M1": 0.5, "M2": 0.4}})

    with pytest.raises(AssertionError) as exc:
        golden_utils.assert_golden("sample", {"results": {"M1": 0.5, "M3": 0.9}})

    msg = str(exc.value)
    assert "results.M2" in msg, f"삭제된 키가 메시지에 없다:\n{msg}"
    assert "results.M3" in msg, f"추가된 키가 메시지에 없다:\n{msg}"


def test_mismatch_message_reports_list_index_and_length(golden_dir):
    """[H-02] 리스트는 길이 변화와 어긋난 인덱스를 함께 지목해야 한다."""
    _seed(golden_dir, "sample", {"warnings": ["a", "b"]})

    with pytest.raises(AssertionError) as exc:
        golden_utils.assert_golden("sample", {"warnings": ["a", "z", "c"]})

    msg = str(exc.value)
    assert "warnings[1]" in msg, f"어긋난 인덱스가 메시지에 없다:\n{msg}"
    assert "len" in msg, f"길이 변화가 메시지에 없다:\n{msg}"


def test_mismatch_message_keeps_update_golden_hint(golden_dir):
    """진단을 바꾸더라도 갱신 방법 안내는 남아 있어야 한다(기존 동작 보존)."""
    _seed(golden_dir, "sample", {"a": 1})

    with pytest.raises(AssertionError) as exc:
        golden_utils.assert_golden("sample", {"a": 2})

    assert "UPDATE_GOLDEN" in str(exc.value)


def test_rounding_tolerance_is_preserved(golden_dir):
    """6자리 반올림 허용 오차는 그대로여야 한다 — 진단 개선이 비교 기준을 바꾸면 안 된다."""
    _seed(golden_dir, "sample", {"v": 0.1234567})

    golden_utils.assert_golden("sample", {"v": 0.12345674})  # 6자리 내 동일 → 통과


def test_bootstrap_still_writes_when_missing(golden_dir):
    """골든이 없으면 생성 후 통과하는 부트스트랩 동작은 유지."""
    golden_utils.assert_golden("brand_new", {"a": 1})

    assert json.loads((golden_dir / "brand_new.json").read_text(encoding="utf-8")) == {"a": 1}


# ── 비결정적 필드 리댁션 (ISSUES.md F-09 로 도입) ──────────────────────────

def test_redaction_keeps_golden_stable_across_runs(tmp_path, monkeypatch):
    """평가 시각·라이브러리 버전이 달라져도 골든이 깨지지 않는다."""
    import golden_utils

    monkeypatch.setattr(golden_utils, "GOLDEN_DIR", tmp_path)
    first = {"results": {"M1": 1.0}, "environment": {"evaluated_at": "2026-01-01T00:00:00+09:00",
                                                    "libraries": {"pandas": "2.2.0"}}}
    golden_utils.assert_golden("redact_case", first)

    second = {"results": {"M1": 1.0}, "environment": {"evaluated_at": "2026-09-07T12:34:56+09:00",
                                                     "libraries": {"pandas": "9.9.9"}}}
    golden_utils.assert_golden("redact_case", second)   # 예외가 나면 실패


def test_redaction_still_catches_a_missing_field(tmp_path, monkeypatch):
    """값은 안 보되 **필드가 통째로 사라지면** 잡아야 한다."""
    import pytest
    import golden_utils

    monkeypatch.setattr(golden_utils, "GOLDEN_DIR", tmp_path)
    golden_utils.assert_golden("redact_missing", {
        "results": {}, "environment": {"evaluated_at": "x", "libraries": {}},
    })

    with pytest.raises(AssertionError):
        golden_utils.assert_golden("redact_missing", {"results": {}})


def test_redaction_does_not_hide_other_changes(tmp_path, monkeypatch):
    """리댁션이 다른 경로의 회귀를 가리면 안 된다."""
    import pytest
    import golden_utils

    monkeypatch.setattr(golden_utils, "GOLDEN_DIR", tmp_path)
    golden_utils.assert_golden("redact_other", {
        "results": {"M1": 1.0}, "environment": {"evaluated_at": "x", "libraries": {}},
    })

    with pytest.raises(AssertionError):
        golden_utils.assert_golden("redact_other", {
            "results": {"M1": 0.5}, "environment": {"evaluated_at": "y", "libraries": {}},
        })


def test_redaction_catches_a_missing_subfield(tmp_path, monkeypatch):
    """리댁션 범위가 너무 넓으면 **하위 필드**가 사라져도 못 잡는다.

    `environment` 를 통째로 자리표시자로 바꾸면 `libraries` 가 빠져도 같은 문자열이라
    비교가 통과한다. 값만 가리고 **구조는 남겨야** 한다.
    """
    import pytest
    import golden_utils

    monkeypatch.setattr(golden_utils, "GOLDEN_DIR", tmp_path)
    golden_utils.assert_golden("redact_subfield", {
        "environment": {"evaluated_at": "x", "libraries": {"pandas": "2.0"}},
    })

    with pytest.raises(AssertionError):
        golden_utils.assert_golden("redact_subfield", {"environment": {"evaluated_at": "y"}})
