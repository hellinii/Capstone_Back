"""
validator.py — 사용자가 확정한 컬럼 매핑의 유효성 검사

task_type별 필수/선택 역할 규칙을 검사하고,
현재 매핑으로 계산 가능한 지표 목록을 반환합니다.

이미지 기준 지표 가용성 규칙:
  [Binary]
    y_true + y_pred        → M1~M8, M20~M23
    y_true + score_positive→ M9, M10, M19
    (y_pred OR score_positive 중 최소 하나 필수)

  [Multiclass]
    y_true + y_pred        → M1~M6, M11~M14, M21~M23

  [Multilabel]
    true_labels + pred_labels          → M1~M6, M11~M13, M15~M17, M21~M23
    true_labels + score_per_label(N개) → M18
"""

from app.core.schemas import ColumnMapping, ColumnRole, TC_REQUIREMENTS, TaskType
from app.analysis.schemas import ConfirmMappingRequest, ConfirmMappingResponse, MappingValidationError, MappingValidationWarning


# ── 지표 가용성 규칙 정의 ────────────────────────────────────────────────────────
# TC_REQUIREMENTS(각 지표 계산에 필요한 role 집합)는 evaluator.engine 과 공유하므로
# app.core.schemas 로 이동했다. 여기서는 import 해서 사용한다.

# task_type별 필수 역할 (없으면 is_valid=False)
_REQUIRED_ROLES: dict[TaskType, list[ColumnRole]] = {
    TaskType.binary: [
        ColumnRole.y_true,
        # y_pred와 score_positive 중 최소 하나 (아래 별도 체크)
    ],
    TaskType.multiclass: [
        ColumnRole.y_true,
        ColumnRole.y_pred,
    ],
    TaskType.multilabel: [
        ColumnRole.true_labels,
        ColumnRole.pred_labels,
    ],
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
            "y_pred가 없어 M1~M8, M20~M23 (예측 기반 지표)를 계산할 수 없습니다.",
        ),
    ],
    TaskType.multiclass: [
        (
            ColumnRole.prob_per_class,
            "MISSING_PROB_PER_CLASS",
            "prob_per_class가 없어 클래스별 확률 기반 세부 지표를 계산할 수 없습니다.",
        ),
    ],
    TaskType.multilabel: [
        (
            ColumnRole.score_per_label,
            "MISSING_SCORE_PER_LABEL",
            "score_per_label이 없어 M18 (Distribution Diff ML)을 계산할 수 없습니다.",
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
    selected_tcs = request.selected_tcs

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
    for required_role in _REQUIRED_ROLES[task_type]:
        if required_role not in mapped_roles:
            errors.append(MappingValidationError(
                code="MISSING_REQUIRED",
                message=f"필수 역할 '{required_role.value}'이 매핑되지 않았습니다.",
            ))

    # Binary 전용: y_pred와 score_positive 중 최소 하나 필수
    if task_type == TaskType.binary:
        has_pred = ColumnRole.y_pred in mapped_roles
        has_score = ColumnRole.score_positive in mapped_roles
        if not has_pred and not has_score:
            errors.append(MappingValidationError(
                code="MISSING_PRED_OR_SCORE",
                message="Binary 평가는 y_pred 또는 score_positive 중 최소 하나가 필요합니다.",
            ))

    # ── 3. 선택 역할 누락 → 경고 ─────────────────────────────────────────────
    for (role, code, message) in _WARNING_CONDITIONS.get(task_type, []):
        if role not in mapped_roles:
            warnings.append(MappingValidationWarning(code=code, message=message))

    # ── 4. 지표 가용성 계산 및 선택된 지표 검증 ────────────────────────────────────────────────────
    tc_requirements = TC_REQUIREMENTS[task_type]
    available_tcs: list[str] = []
    unavailable_tcs: list[str] = []

    for tc_name, required_roles in sorted(tc_requirements.items(), key=lambda x: _tc_sort_key(x[0])):
        if required_roles.issubset(mapped_roles):
            available_tcs.append(tc_name)
        else:
            missing = required_roles - mapped_roles
            missing_str = ", ".join(r.value for r in missing)
            unavailable_tcs.append(f"{tc_name} (누락: {missing_str})")
            
            # 🔥 [변경점] 사용자가 계산하겠다고 명시적으로 선택한 지표인데, 필요 역할이 매핑되지 않았다면 Error 처리
            if tc_name in selected_tcs:
                errors.append(MappingValidationError(
                    code="MISSING_TC_REQUIREMENT",
                    message=f"선택하신 지표 '{tc_name}'를 계산하려면 [{missing_str}] 역할의 컬럼 매핑이 필수입니다."
                ))

    is_valid = len(errors) == 0

    return ConfirmMappingResponse(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        available_tcs=available_tcs,
        unavailable_tcs=unavailable_tcs,
        confirmed_mappings=mappings,
    )


def _tc_sort_key(tc_name: str) -> int:
    """'M1', 'M23' 같은 지표 ID 문자열을 숫자 기준으로 정렬하기 위한 키(접두사 무관)."""
    return int("".join(c for c in tc_name if c.isdigit()))
