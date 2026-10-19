"""자동 매핑 AI 지시문(app/analysis/prompt_builder.py) — M1.

다중·멀티 확률 컬럼을 ignore 로 보내라고 안내하던 지시문을 고쳤다. 확률 역할은 세 유형 모두
정식 입력이라(2026-09-07 결정 1) y_pred 가 없으면 확률에서 예측을 파생한다. 지시문이 확률을
ignore 로 보내면 평가할 예측이 사라진다.
"""
import pytest

from app.analysis.prompt_builder import _ROLE_HINTS, build_system_prompt, role_hints_text
from app.core.schemas import ColumnRole, TaskType, VALID_ROLES_BY_TASK


@pytest.mark.parametrize("task_type", list(TaskType))
def test_hint_roles_equal_valid_roles(task_type):
    """유형마다 지시문이 설명하는 역할 집합 = 허용 역할 집합(비어 있지 않음 먼저)."""
    hinted = set(_ROLE_HINTS[task_type])
    assert hinted, "힌트가 비어 있다"
    assert hinted == set(VALID_ROLES_BY_TASK[task_type])


@pytest.mark.parametrize("task_type", list(TaskType))
def test_prompt_never_sends_probability_columns_to_ignore(task_type):
    prompt = build_system_prompt(task_type)
    assert "확률 컬럼도" not in prompt
    ignore_line = next(l for l in prompt.splitlines() if l.startswith("ignore:"))
    assert "확률" not in ignore_line


def test_multiclass_prompt_names_prob_per_class_with_column_pattern():
    text = role_hints_text(TaskType.multiclass)
    line = next(l for l in text.splitlines() if l.startswith(f"{ColumnRole.prob_per_class.value}:"))
    assert "prob_<클래스명>" in line and "0~1" in line


def test_multilabel_prompt_names_score_per_label_with_column_pattern():
    text = role_hints_text(TaskType.multilabel)
    line = next(l for l in text.splitlines() if l.startswith(f"{ColumnRole.score_per_label.value}:"))
    assert "score_<레이블명>" in line and "prob_<레이블명>" in line
