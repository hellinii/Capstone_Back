"""app/evaluation/service.py — 평가 파이프라인 오케스트레이션

파싱된 DataFrame 과 요청(EvaluateRequest)을 받아 컬럼 충돌 검사 → 매핑 변환 →
지표 계산(engine.evaluate) → 전처리 오류 감지 → 리포트 조립을 수행하고 EvaluateResponse 를
반환한다. 실패는 EvaluationError(code) 로 표현하며 HTTP 상태코드 매핑은 라우터가 담당한다.
(구 evaluate_dataset 라우터의 인라인 오케스트레이션에서 추출, 동작 불변.)

상호작용
- 의존(import): app.core.schemas(EvaluateRequest/Response), app.analysis.validator(find_column_conflicts),
  app.evaluation.engine(evaluate), app.evaluation.report(generate_report)
- 사용처: app.evaluation.router.evaluate_dataset
"""
from app.evaluation.schemas import EvaluateRequest, EvaluateResponse, EvaluationEnvironment
from app.evaluation.environment import evaluated_at, library_versions
from app.analysis.validator import find_column_conflicts, find_invalid_roles
from app.evaluation.engine import evaluate as run_evaluation
from app.evaluation.report import generate_report
from app.evaluation.labels import normalize_distribution


class EvaluationError(Exception):
    """평가 파이프라인 실패. code: mapping_conflict | compute_error | preprocess_error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def run_evaluation_pipeline(df, request: EvaluateRequest) -> EvaluateResponse:
    """충돌검사 → 매핑변환 → engine.evaluate → 전처리오류 → 리포트 조립 → EvaluateResponse."""
    # task_type 에 허용되지 않는 역할 차단 (SPEC §4, ISSUES.md A-10).
    # confirm-mapping 을 건너뛰고 이 엔드포인트를 직접 호출하는 경로의 백스톱이다 —
    # 프론트 드롭다운 제한은 UI 경로만 막는다.
    invalid_roles = find_invalid_roles(request.column_mappings, request.task_type)
    if invalid_roles:
        raise EvaluationError("mapping_conflict", "; ".join(e.message for e in invalid_roles))

    # 컬럼 매핑 충돌 검사 (정답=예측 동일 컬럼 등 → 가짜 100% 차단)
    conflicts = find_column_conflicts(request.column_mappings, request.task_type)
    if conflicts:
        raise EvaluationError("mapping_conflict", "; ".join(c.message for c in conflicts))

    mappings = [{"column": m.column, "role": m.role.value} for m in request.column_mappings]
    positive_class = request.metadata.positive_class
    beta = request.beta

    try:
        eval_results = run_evaluation(
            df=df,
            mappings=mappings,
            task_type=request.task_type.value,
            selected_metric_ids=request.selected_metric_ids,
            positive_class=positive_class,
            beta=beta,
            decision_threshold=request.decision_threshold,
            metadata=request.metadata.model_dump(),
        )
    except Exception as e:
        raise EvaluationError("compute_error", f"평가 연산 실행 오류: {str(e)}")

    # 전처리 에러 확인
    if "error" in eval_results:
        raise EvaluationError("preprocess_error", eval_results["error"])

    metadata = eval_results.pop("_metadata", {})
    warnings = metadata.get("warnings", [])
    dropped_rows = metadata.get("dropped_rows", 0)
    n_samples = metadata.get("n_samples", 0)
    # 표현형·순서 정규화는 labels.normalize_distribution 이 정본이다(D-04).
    # 종전 `{str(k): v}` 는 int 1 과 str '1' 을 뭉개며 카운트를 덮어썼다.
    class_distribution = normalize_distribution(metadata.get("class_distribution", {}))

    formatted_results = generate_report(eval_results)

    return EvaluateResponse(
        results=formatted_results,
        # 성적서 4절이 인쇄하는 '평가 도구·일시'를 서버가 확정한다(ISSUES.md F-09).
        environment=EvaluationEnvironment(
            libraries=library_versions(), evaluated_at=evaluated_at()
        ),
        warnings=warnings,
        dropped_rows=dropped_rows,
        n_samples=n_samples,
        class_distribution=class_distribution,
    )
