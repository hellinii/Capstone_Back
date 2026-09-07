"""tests/test_fallback_notices.py — 규칙 폴백도 안내를 만든다.

ISSUES.md B-03 (2026-09-07 ★작은 판단 ① — "규칙 폴백에도 안내를 만들도록 백엔드를
함께 고친다 — 지금은 컬럼이 조용히 사라지는 순간에 정확히 안내가 비어 있다").

LLM 경로(`reconcile`)는 오히려 안내를 성실히 만든다. **정작 비어 있던 것은 규칙
폴백**이다 — `else` 없는 elif 사슬이라 어느 규칙에도 걸리지 않은 컬럼이 조용히
`ignore` 가 되고 `column_notes` 는 `[]` 였다. 무키 환경(OPENAI_API_KEY 미설정)이나
예산 소진 시 사용자는 **가장 안내가 필요한 순간에 아무 설명도 받지 못했다.**
"""
import pandas as pd

from app.analysis.fallback_mapper import analyze_columns_fallback
from app.core.schemas import ColumnRole, TaskType


def _run(columns, task_type=TaskType.binary):
    df = pd.DataFrame({c: [1, 0] for c in columns})
    return analyze_columns_fallback(task_type=task_type, columns=columns, df=df)


def test_unmapped_columns_produce_a_notice():
    """평범한 헤더가 전부 ignore 로 강등되면 그 사실을 알린다."""
    resp = _run(["sample", "gold", "guess", "conf", "ms"])

    assert all(m.role == ColumnRole.ignore for m in resp.column_mappings)
    assert resp.column_notes, "전 컬럼이 ignore 인데 안내가 비어 있다"

    unmapped = [n for n in resp.column_notes if n.status == "unmapped_header"]
    assert {n.llm_column for n in unmapped} == {"sample", "gold", "guess", "conf", "ms"}


def test_notice_says_the_column_is_excluded_from_evaluation():
    resp = _run(["memo"])
    note = next(n for n in resp.column_notes if n.llm_column == "memo")

    assert "평가" in note.message
    assert note.matched_column is None


def test_mapped_columns_do_not_produce_a_notice():
    """규칙에 걸린 컬럼은 조용히 넘어간다 — 안내가 소음이 되면 안 읽힌다."""
    resp = _run(["id", "y_true", "y_pred"])

    assert resp.column_notes == []


def test_notice_marks_the_fallback_as_the_source():
    """LLM 이 아니라 규칙으로 매핑됐다는 사실을 알린다."""
    resp = _run(["y_true", "y_pred", "junk"])

    assert any("규칙" in n.message for n in resp.column_notes)


def test_fallback_does_not_assign_a_role_the_task_forbids():
    """[A-10] 폴백이 VALID_ROLES_BY_TASK 를 어기면 그 매핑이 곧바로 거절된다.

    실측으로 multiclass 폴백이 `prob_per_class` 를 배정했는데, 결정 1 이전에는 그것이
    허용 역할이 아니었다 — 폴백 결과가 자기 시스템의 검증을 통과하지 못했다.
    """
    from app.core.schemas import VALID_ROLES_BY_TASK

    for task in TaskType:
        resp = _run(["id", "y_true", "y_pred", "probability_cat", "score_x", "latency_ms"], task)
        allowed = set(VALID_ROLES_BY_TASK[task])
        offenders = [m for m in resp.column_mappings if m.role not in allowed]
        assert offenders == [], f"{task}: {[(m.column, m.role) for m in offenders]}"
