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
    true_labels + pred_labels  → M1~M5, M11~M13, M15~M18, M21, M22
    (score_per_label 을 요구하는 지표는 없다 — 확률 범위 검증용 선택 컬럼)
"""

from app.core.schemas import ColumnMapping, ColumnRole, METRIC_REQUIREMENTS, TaskType
from app.analysis.schemas import ConfirmMappingRequest, ConfirmMappingResponse, MappingValidationError, MappingValidationWarning


# ── 지표 가용성 규칙 정의 ────────────────────────────────────────────────────────
# METRIC_REQUIREMENTS(각 지표 계산에 필요한 role 집합)는 evaluator.engine 과 공유하므로
# app.core.schemas 로 이동했다. 여기서는 import 해서 사용한다.

# task_type별 "정답" 역할 — 어떤 지표를 고르든 항상 필수 (없으면 is_valid=False)
_TRUTH_ROLE: dict[TaskType, ColumnRole] = {
    TaskType.binary:     ColumnRole.y_true,
    TaskType.multiclass: ColumnRole.y_true,
    TaskType.multilabel: ColumnRole.true_labels,
}

# task_type별 "예측" 역할 후보 — 하나라도 매핑돼 있으면 충족.
# 단, 선택한 지표가 모두 예측 역할을 쓰지 않으면(예: M23 만 선택) 이 요구는 면제된다.
# 면제 여부는 METRIC_REQUIREMENTS 에서 파생시키므로 특정 지표 ID 를 하드코딩하지 않는다.
_PRED_ROLES: dict[TaskType, list[ColumnRole]] = {
    TaskType.binary:     [ColumnRole.y_pred, ColumnRole.score_positive],
    TaskType.multiclass: [ColumnRole.y_pred],
    TaskType.multilabel: [ColumnRole.pred_labels],
}

# task_type별 경고 조건: (없을 때 경고할 role, 경고 코드, 메시지)
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
            "y_pred가 없어 M1~M8, M20~M22 (예측 기반 지표)를 계산할 수 없습니다.",
        ),
    ],
    TaskType.multiclass: [
        (
            ColumnRole.prob_per_class,
            "MISSING_PROB_PER_CLASS",
            "prob_per_class가 없어 클래스별 확률 기반 세부 지표를 계산할 수 없습니다.",
        ),
    ],
    # multilabel: score_per_label 을 요구하는 지표가 없다(M18 도 true_labels/pred_labels 기반).
    # 확률 범위 검증(preprocessor._validate_score_columns)에만 쓰이는 선택 컬럼이라 경고하지 않는다.
    TaskType.multilabel: [],
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


def validate_mapping(request: ConfirmMappingRequest) -> ConfirmMappingResponse:
    """
    사용자가 확정한 매핑을 검증하고 계산 가능한 지표 목록을 반환합니다.

    검사 순서:
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

    # ── 1. 역할 중복 체크 (ignore/score_per_class처럼 여러 개 허용되는 것 제외) ──
    _MULTI_ALLOWED = {
        ColumnRole.ignore,
        ColumnRole.prob_per_class,
        ColumnRole.score_per_label,
    }
    role_counts: dict[ColumnRole, int] = {}
    for m in mappings:
        role_counts[m.role] = role_counts.get(m.role, 0) + 1

    for role, count in role_counts.items():
        if count > 1 and role not in _MULTI_ALLOWED:
            errors.append(MappingValidationError(
                code="DUPLICATE_ROLE",
                message=f"'{role.value}' 역할이 {count}개 컬럼에 중복 매핑되어 있습니다. 하나만 지정해주세요.",
            ))

    # ── 1-2. 컬럼 단위 상호배타 체크 (정답=예측 동일 컬럼 등) ────────────────────
    errors.extend(find_column_conflicts(mappings, task_type))

    # ── 2. 필수 역할 누락 체크 ────────────────────────────────────────────────
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

    # ── 3. 선택 역할 누락 → 경고 ─────────────────────────────────────────────
    for (role, code, message) in _WARNING_CONDITIONS.get(task_type, []):
        if role not in mapped_roles:
            warnings.append(MappingValidationWarning(code=code, message=message))

    # ── 4. 지표 가용성 계산 및 선택된 지표 검증 ────────────────────────────────────────────────────
    # metric_requirements 는 위 2단계에서 이미 조회했다.
    available_metric_ids: list[str] = []
    unavailable_metric_ids: list[str] = []

    for metric_id, required_roles in sorted(metric_requirements.items(), key=lambda x: _metric_sort_key(x[0])):
        if required_roles.issubset(mapped_roles):
            available_metric_ids.append(metric_id)
        else:
            missing = required_roles - mapped_roles
            missing_str = ", ".join(r.value for r in missing)
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


def _metric_sort_key(metric_id: str) -> int:
    """'M1', 'M23' 같은 지표 ID 문자열을 숫자 기준으로 정렬하기 위한 키(접두사 무관)."""
    return int("".join(c for c in metric_id if c.isdigit()))
