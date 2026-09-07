"""
validator.py — 사용자가 확정한 컬럼 매핑의 유효성 검사

task_type별 필수/선택 역할 규칙을 검사하고,
현재 매핑으로 계산 가능한 지표 목록을 반환합니다.

정답 역할(y_true/true_labels)은 항상 필수이고, 예측 역할은 "선택한 지표 중 하나라도
예측을 쓸 때"만 필수입니다. 예컨대 M23(Imbalance Ratio)만 선택하면 정답 분포만으로
계산되므로 예측 컬럼 없이도 진행할 수 있습니다. selected_metric_ids 를 보내지 않는
호출자는 어떤 지표가 필요한지 알 수 없으므로 종전대로 예측 역할을 필수로 유지합니다.

지표 가용성 규칙(app.core.schemas.METRIC_REQUIREMENTS 와 동일한 내용):
  [Binary]
    y_true                 → M23
    y_true + y_pred        → M1~M8, M20~M22
    y_true + score_positive→ M9, M10, M19
    (y_pred OR score_positive 중 최소 하나 필수)

  [Multiclass]
    y_true                 → M23
    y_true + y_pred        → M1~M6, M11~M14, M21, M22

  [Multilabel]
    true_labels                → M23
    true_labels + pred_labels  → M2~M5, M15~M18, M21, M22
    (M1·M11·M12·M13 은 값이 M16·M2~M4·M22 와 겹쳐 노출하지 않는다 — 결정 2)
    (score_per_label 은 pred_labels 의 대체 입력이다 — 없으면 임계값으로 파생한다)
"""

from app.core.schemas import (
    ColumnMapping,
    ColumnRole,
    METRIC_REQUIREMENTS,
    MULTI_COLUMN_ROLES,
    PREDICTION_ROLES_BY_TASK,
    TRUTH_ROLE_BY_TASK,
    TaskType,
    VALID_ROLES_BY_TASK,
)
from app.analysis.schemas import ConfirmMappingRequest, ConfirmMappingResponse, MappingValidationError, MappingValidationWarning


# ── 지표 가용성 규칙 정의 ────────────────────────────────────────────────────────
# METRIC_REQUIREMENTS(각 지표 계산에 필요한 role 집합)와 예측 역할의 대체 규칙
# (PREDICTION_ROLES_BY_TASK)은 evaluator.engine·preprocessor 와 공유하므로
# app.core.schemas 가 단일 출처다. 여기서는 import 해서 사용한다.

_TRUTH_ROLE = TRUTH_ROLE_BY_TASK

# task_type별 "예측" 역할 후보 — 주 역할 또는 그것을 파생할 수 있는 확률 역할.
# 단, 선택한 지표가 모두 예측 역할을 쓰지 않으면(예: M23 만 선택) 이 요구는 면제된다.
# 면제 여부는 METRIC_REQUIREMENTS 에서 파생시키므로 특정 지표 ID 를 하드코딩하지 않는다.
_PRED_ROLES: dict[TaskType, list[ColumnRole]] = {
    task: [primary, *alternatives]
    for task, (primary, alternatives) in PREDICTION_ROLES_BY_TASK.items()
}

# task_type별 경고 조건: (없을 때 경고할 role, 경고 코드, 메시지)
#
# 문구는 **사실이어야 한다.** 종전 multiclass 의 MISSING_PROB_PER_CLASS 는 "확률 기반 세부
# 지표를 계산할 수 없다"고 했으나 multiclass 에는 확률을 읽는 지표가 하나도 없어 정상 매핑
# 에서도 항상 뜨는 거짓 경고였다(ISSUES.md A-11). 확률의 실제 쓸모는 '예측의 대체 입력'이다.
_WARNING_CONDITIONS: dict[TaskType, list[tuple[ColumnRole, str, str]]] = {
    TaskType.binary: [
        (
            ColumnRole.score_positive,
            "MISSING_SCORE_POSITIVE",
            "score_positive가 없어 M9, M10, M19 (확률 기반 지표)를 계산할 수 없습니다.",
        ),
        (
            ColumnRole.y_pred,
            "MISSING_Y_PRED",
            "y_pred가 없어 score_positive와 결정 임계값으로 예측을 파생합니다. "
            "파생값은 모델의 실제 출력이 아니며 성적서에 그 사실이 기재됩니다.",
        ),
    ],
    TaskType.multiclass: [
        (
            ColumnRole.y_pred,
            "MISSING_Y_PRED",
            "y_pred가 없어 prob_per_class의 argmax로 예측을 파생합니다. "
            "파생값은 모델의 실제 출력이 아니며 성적서에 그 사실이 기재됩니다.",
        ),
    ],
    TaskType.multilabel: [
        (
            ColumnRole.pred_labels,
            "MISSING_PRED_LABELS",
            "pred_labels가 없어 score_per_label과 레이블별 결정 임계값으로 예측을 파생합니다. "
            "파생값은 모델의 실제 출력이 아니며 성적서에 그 사실이 기재됩니다.",
        ),
    ],
}


# (y_true & y_pred) 또는 (true_labels & pred_labels)가 같은 컬럼에 매핑되면
# 정답=예측이 되어 모든 지표가 가짜 100%(낮을수록 좋은 지표는 0)로 산출된다.
_TRUE_PRED_PAIRS: list[tuple[ColumnRole, ColumnRole]] = [
    (ColumnRole.y_true, ColumnRole.y_pred),
    (ColumnRole.true_labels, ColumnRole.pred_labels),
]


def find_column_conflicts(
    mappings: list[ColumnMapping], task_type: TaskType
) -> list[MappingValidationError]:
    """한 컬럼이 서로 다른 non-ignore 역할에 동시 배정됐는지 검사한다.

    특히 정답/예측 쌍이 같은 컬럼이면 '가짜 100%' 위험이므로 전용 코드로 보고한다.
    이 헬퍼는 confirm-mapping / validate-data / evaluate 세 경로가 공유한다.
    (동일 컬럼 + 동일 역할 중복은 역할 집합 크기가 1이라 여기서 건너뛰고,
     기존 DUPLICATE_ROLE 검사가 처리한다 → 이중 보고 없음.)
    """
    errors: list[MappingValidationError] = []
    column_to_roles: dict[str, set[ColumnRole]] = {}
    for m in mappings:
        if m.role == ColumnRole.ignore:
            continue
        column_to_roles.setdefault(m.column, set()).add(m.role)

    for column, roles in column_to_roles.items():
        if len(roles) < 2:
            continue
        is_true_pred = any(t in roles and p in roles for (t, p) in _TRUE_PRED_PAIRS)
        if is_true_pred:
            errors.append(MappingValidationError(
                code="SAME_COLUMN_TRUE_PRED",
                message=(
                    f"정답과 예측에 동일한 컬럼 '{column}'이 매핑되었습니다. "
                    "이 경우 예측이 정답과 같아져 정확도 등 모든 지표가 100%로 계산되어 "
                    "평가가 무의미합니다. 서로 다른 컬럼을 지정해주세요."
                ),
            ))
        else:
            role_list = ", ".join(sorted(r.value for r in roles))
            errors.append(MappingValidationError(
                code="COLUMN_MULTIPLE_ROLES",
                message=(
                    f"컬럼 '{column}'이 여러 역할({role_list})에 동시에 매핑되었습니다. "
                    "각 컬럼은 하나의 역할만 가질 수 있습니다."
                ),
            ))
    return errors


def find_invalid_roles(
    mappings: list[ColumnMapping], task_type: TaskType
) -> list[MappingValidationError]:
    """task_type 에 허용되지 않는 역할이 매핑됐는지 검사한다 (SPEC §4, ISSUES.md A-10).

    SPEC §4 는 "Binary 인데 prob_class_*", "Multiclass 인데 score", "Multilabel 인데
    score 또는 prob_class_*" 세 가지를 진행 차단 Error 로 규정한다. 세 가지를 손으로
    나열하는 대신 `VALID_ROLES_BY_TASK` 를 유일 기준으로 삼아 일반화한다 — 규칙 사본을
    하나 더 만들지 않기 위해서다(뿌리 ①). 그러면 SPEC 이 열거하지 않은 조합
    (예: binary 에 true_labels)도 같은 이유로 함께 막힌다.

    **왜 무시로 끝내면 안 되는가.** `frame.required_columns` 가 ignore 가 아닌 모든
    역할의 컬럼을 dropna 대상에 넣는다. 잘못된 역할로 매핑된 컬럼의 결측이 평가 표본을
    조용히 깎으므로, 사용자는 자기가 지정한 역할이 무시됐다는 사실도, 표본이 줄었다는
    사실도 모른 채 성적서를 받는다.

    confirm-mapping 과 evaluate·validate-data 세 경로가 공유한다 — 프론트 드롭다운
    제한은 UI 경로만 막고, API 를 직접 호출하면 그대로 통과했다.
    """
    allowed = set(VALID_ROLES_BY_TASK[task_type])
    errors: list[MappingValidationError] = []
    for m in mappings:
        if m.role in allowed:
            continue
        errors.append(MappingValidationError(
            code="INVALID_ROLE_FOR_TASK",
            message=(
                f"'{task_type.value}' 평가에서는 '{m.role.value}' 역할을 사용할 수 없습니다"
                f"(컬럼 '{m.column}'). 사용 가능한 역할: "
                f"{', '.join(r.value for r in VALID_ROLES_BY_TASK[task_type])}."
            ),
        ))
    return errors


def validate_mapping(request: ConfirmMappingRequest) -> ConfirmMappingResponse:
    """
    사용자가 확정한 매핑을 검증하고 계산 가능한 지표 목록을 반환합니다.

    검사 순서(코드의 번호 주석과 일치한다):
    1. 역할 유효성: task_type에 허용되지 않는 role 사용 여부
    2. 중복 체크: 단일 역할(y_true, y_pred 등)에 여러 컬럼 매핑 여부
    3. 필수 역할 누락 체크 → is_valid 결정
    4. 선택 역할 누락 체크 → warnings
    5. 지표 가용성 계산
    """
    task_type = request.task_type
    mappings = request.column_mappings
    selected_metric_ids = request.selected_metric_ids

    errors: list[MappingValidationError] = []
    warnings: list[MappingValidationWarning] = []

    # 현재 매핑에서 어떤 role들이 사용됐는지 추출
    mapped_roles = {m.role for m in mappings}

    # ── 1. 역할 유효성 — task_type 에 허용되지 않는 역할 (SPEC §4, ISSUES.md A-10) ──
    # 중복 체크보다 **앞**에 둔다. 잘못된 역할을 두 컬럼에 준 경우 사용자가 고쳐야 할 것은
    # 중복이 아니라 역할 자체인데, 뒤에 두면 DUPLICATE_ROLE 이 먼저 보고돼 안내가 어긋난다.
    errors.extend(find_invalid_roles(mappings, task_type))

    # ── 2. 역할 중복 체크 (ignore/확률 역할처럼 여러 개 허용되는 것 제외) ──
    _MULTI_ALLOWED = MULTI_COLUMN_ROLES
    role_counts: dict[ColumnRole, int] = {}
    for m in mappings:
        role_counts[m.role] = role_counts.get(m.role, 0) + 1

    for role, count in role_counts.items():
        if count > 1 and role not in _MULTI_ALLOWED:
            errors.append(MappingValidationError(
                code="DUPLICATE_ROLE",
                message=f"'{role.value}' 역할이 {count}개 컬럼에 중복 매핑되어 있습니다. 하나만 지정해주세요.",
            ))

    # ── 2-1. 컬럼 단위 상호배타 체크 (정답=예측 동일 컬럼 등) ────────────────────
    errors.extend(find_column_conflicts(mappings, task_type))

    # ── 3. 필수 역할 누락 체크 ────────────────────────────────────────────────
    metric_requirements = METRIC_REQUIREMENTS[task_type]
    pred_roles = _PRED_ROLES[task_type]

    # 정답 역할은 무조건 필수
    truth_role = _TRUTH_ROLE[task_type]
    if truth_role not in mapped_roles:
        errors.append(MappingValidationError(
            code="MISSING_REQUIRED",
            message=f"필수 역할 '{truth_role.value}'이 매핑되지 않았습니다.",
        ))

    # 예측 역할은 "선택한 지표 중 하나라도 예측을 쓸 때"만 필수.
    # selected_metric_ids 는 기본값 [] (app.analysis.schemas) 이므로, 지표를 명시하지 않는
    # 호출자는 종전과 동일하게 예측 역할을 필수로 유지한다(가용 지표를 알 수 없으므로 엄격).
    known_selected = [m for m in selected_metric_ids if m in metric_requirements]
    pred_required = not known_selected or any(
        any(role in pred_roles for role in metric_requirements[m]) for m in known_selected
    )

    if pred_required and not (set(pred_roles) & mapped_roles):
        if task_type == TaskType.binary:
            errors.append(MappingValidationError(
                code="MISSING_PRED_OR_SCORE",
                message="Binary 평가는 y_pred 또는 score_positive 중 최소 하나가 필요합니다.",
            ))
        else:
            errors.append(MappingValidationError(
                code="MISSING_REQUIRED",
                message=f"필수 역할 '{pred_roles[0].value}'이 매핑되지 않았습니다.",
            ))

    # ── 4. 선택 역할 누락 → 경고 ─────────────────────────────────────────────
    for (role, code, message) in _WARNING_CONDITIONS.get(task_type, []):
        if role not in mapped_roles:
            warnings.append(MappingValidationWarning(code=code, message=message))

    # ── 5. 지표 가용성 계산 및 선택된 지표 검증 ────────────────────────────────────────────────────
    # metric_requirements 는 위 3단계에서 이미 조회했다.
    available_metric_ids: list[str] = []
    unavailable_metric_ids: list[str] = []

    for metric_id, required_roles in sorted(metric_requirements.items(), key=lambda x: _metric_sort_key(x[0])):
        missing = _unsatisfied_roles(required_roles, mapped_roles, task_type)
        if not missing:
            available_metric_ids.append(metric_id)
        else:
            # 순서를 정렬로 고정한다 — str-Enum set 의 iteration 순서는 PYTHONHASHSEED 에
            # 따라 달라져 같은 입력이 실행마다 다른 문구를 만들었다(ISSUES.md H-07).
            missing_str = ", ".join(sorted(r.value for r in missing))
            unavailable_metric_ids.append(f"{metric_id} (누락: {missing_str})")
            
            # 🔥 [변경점] 사용자가 계산하겠다고 명시적으로 선택한 지표인데, 필요 역할이 매핑되지 않았다면 Error 처리
            if metric_id in selected_metric_ids:
                errors.append(MappingValidationError(
                    code="MISSING_METRIC_REQUIREMENT",
                    message=f"선택하신 지표 '{metric_id}'를 계산하려면 [{missing_str}] 역할의 컬럼 매핑이 필수입니다."
                ))

    is_valid = len(errors) == 0

    return ConfirmMappingResponse(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        available_metric_ids=available_metric_ids,
        unavailable_metric_ids=unavailable_metric_ids,
        confirmed_mappings=mappings,
    )


def _unsatisfied_roles(
    required_roles: set[ColumnRole], mapped_roles: set[ColumnRole], task_type: TaskType
) -> set[ColumnRole]:
    """요구 역할 중 매핑으로 충족되지 않은 것.

    예측 역할(y_pred/pred_labels)은 **확률 역할로도 충족된다** — 하드 예측이 없으면
    전처리가 확률에서 파생하기 때문이다(PREDICTION_ROLES_BY_TASK, ISSUES.md A-01·A-02).
    종전에는 단순 issubset 이라 확률만 제출한 사용자가 `MISSING_METRIC_REQUIREMENT` 로
    막혔다 — SPEC §5 가 '계산 가능'이라고 적은 조합인데도.
    """
    primary, alternatives = PREDICTION_ROLES_BY_TASK[task_type]
    unsatisfied = set()
    for role in required_roles:
        if role in mapped_roles:
            continue
        if role == primary and any(alt in mapped_roles for alt in alternatives):
            continue
        unsatisfied.add(role)
    return unsatisfied


def _metric_sort_key(metric_id: str) -> int:
    """'M1', 'M23' 같은 지표 ID 문자열을 숫자 기준으로 정렬하기 위한 키(접두사 무관)."""
    return int("".join(c for c in metric_id if c.isdigit()))
