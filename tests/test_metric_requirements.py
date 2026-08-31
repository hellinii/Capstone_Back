"""METRIC_REQUIREMENTS(선언) ↔ metrics/*(실제 구현) 정합성 계약 테스트.

METRIC_REQUIREMENTS 는 confirm-mapping 이 "이 지표를 계산하려면 어떤 컬럼이 필요한가"를
사용자에게 강제하는 유일한 근거인데(app/analysis/validator.py 의 MISSING_METRIC_REQUIREMENT),
정작 실제 계산 함수와 어긋나도 아무 테스트가 깨지지 않았다. 두 방향 모두 사용자를 막는다.

  - 과소 선언: 표가 요구하지 않은 역할을 계산 함수가 읽음
      → confirm-mapping 은 통과시키는데 evaluate 에서 실패
  - 과다 선언: 계산 함수가 읽지도 않는 역할을 표가 요구함
      → 계산 가능한 지표인데 사용자를 매핑 단계에서 차단

실제로 M18(과소 선언)과 M23(과다 선언 ×3)이 이 상태로 배포되어 있었다.
"""

import pandas as pd
import pytest

from app.core.schemas import METRIC_REQUIREMENTS, ColumnRole, TaskType
from app.evaluation.engine import METRIC_REGISTRY

# 역할 → (컬럼명, 값). 모든 지표가 계산 가능한 최소 합성 데이터셋.
_COLUMNS: dict[TaskType, dict[ColumnRole, tuple[str, list]]] = {
    TaskType.binary: {
        ColumnRole.y_true:         ("t", [1, 0, 1, 0, 1, 0]),
        ColumnRole.y_pred:         ("p", [1, 0, 0, 0, 1, 1]),
        ColumnRole.score_positive: ("s", [0.9, 0.2, 0.4, 0.1, 0.8, 0.6]),
    },
    TaskType.multiclass: {
        ColumnRole.y_true: ("t", ["a", "b", "c", "a", "b", "c"]),
        ColumnRole.y_pred: ("p", ["a", "b", "a", "a", "c", "c"]),
    },
    TaskType.multilabel: {
        ColumnRole.true_labels:     ("t", [["a"], ["a", "b"], ["b"], ["c"], ["a", "c"], ["b"]]),
        ColumnRole.pred_labels:     ("p", [["a"], ["a"], ["b", "c"], ["c"], ["a"], ["b"]]),
        ColumnRole.score_per_label: ("s", [0.9, 0.8, 0.1, 0.2, 0.7, 0.3]),
    },
}

# 계산 함수가 mapping_dict 에서 읽는 내부 파라미터. engine.evaluate(engine.py:90-92)가
# 주입하는 것과 동일하게 맞춘다.
_EXTRA: dict[TaskType, dict] = {
    TaskType.binary:     {"_task_type": "binary",     "_positive_class": 1,    "_beta": 1.0},
    TaskType.multiclass: {"_task_type": "multiclass", "_positive_class": None, "_beta": 1.0},
    TaskType.multilabel: {"_task_type": "multilabel", "_positive_class": None, "_beta": 1.0},
}

_CASES = [(task, metric_id) for task, reqs in METRIC_REQUIREMENTS.items() for metric_id in sorted(reqs)]
_IDS = [f"{task.value}-{metric_id}" for task, metric_id in _CASES]


def _frame_and_mapping(task: TaskType, roles) -> tuple[pd.DataFrame, dict]:
    """주어진 역할 집합만 매핑된 DataFrame + mapping_dict 를 만든다."""
    data: dict[str, list] = {}
    mapping: dict = {}
    for role in roles:
        column, values = _COLUMNS[task][role]
        data[column] = values
        mapping[role.value] = column
    mapping.update(_EXTRA[task])
    return pd.DataFrame(data), mapping


@pytest.mark.parametrize("task,metric_id", _CASES, ids=_IDS)
def test_declared_roles_are_sufficient(task: TaskType, metric_id: str):
    """표가 선언한 역할만 매핑해도 실제 계산 함수가 동작해야 한다(과소 선언 탐지).

    실패하면 표가 실제보다 적게/다르게 요구하고 있다는 뜻이며,
    사용자는 confirm-mapping 을 통과한 뒤 evaluate 에서 에러를 만난다.
    """
    roles = METRIC_REQUIREMENTS[task][metric_id]
    df, mapping = _frame_and_mapping(task, roles)

    try:
        METRIC_REGISTRY[metric_id](df, mapping)
    except Exception as e:
        declared = ", ".join(sorted(r.value for r in roles))
        pytest.fail(
            f"{task.value} {metric_id}: 표가 선언한 역할 [{declared}] 만으로는 계산이 실패한다 "
            f"({type(e).__name__}: {e}). 표가 실제 구현보다 적게 요구하고 있다."
        )


@pytest.mark.parametrize("task,metric_id", _CASES, ids=_IDS)
def test_declared_roles_are_necessary(task: TaskType, metric_id: str):
    """표가 선언한 역할 중 하나라도 빼면 계산이 실패해야 한다(과다 선언 탐지).

    실패하면 계산에 쓰이지도 않는 컬럼을 사용자에게 강제하고 있다는 뜻이다.
    """
    roles = METRIC_REQUIREMENTS[task][metric_id]

    for dropped in sorted(roles, key=lambda r: r.value):
        df, mapping = _frame_and_mapping(task, roles - {dropped})
        try:
            METRIC_REGISTRY[metric_id](df, mapping)
        except Exception:
            continue  # 기대한 동작: 없으면 계산 불가
        declared = ", ".join(sorted(r.value for r in roles))
        pytest.fail(
            f"{task.value} {metric_id}: 표는 [{declared}] 를 요구하지만 "
            f"'{dropped.value}' 없이도 계산된다. 실제로 읽지 않는 역할을 강제하고 있다."
        )
