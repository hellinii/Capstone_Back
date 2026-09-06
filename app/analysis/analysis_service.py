"""app/analysis/analysis_service.py — 컬럼 매핑 전략 오케스트레이션

"어떤 매핑 전략을 쓸지" 정책을 소유한다: 키 없음 → 규칙 폴백, LLM 호출 실패 → 규칙 폴백으로
graceful degrade, 폴백까지 실패하면 도메인 예외(AnalysisError). HTTP 를 모르며 상태코드
매핑은 라우터가 담당한다. (구 validation/analysis router 인라인 정책에서 추출.)

상호작용
- 의존(import): app.analysis.llm_mapper(analyze_columns_with_llm),
  app.analysis.fallback_mapper(analyze_columns_fallback), app.core.schemas(AnalysisResponse, TaskType)
- 사용처: app.analysis.router.analyze_columns
"""
import logging

from openai import AsyncOpenAI

from app.core.schemas import TaskType
from app.analysis.schemas import AnalysisResponse
from app.analysis.llm_mapper import analyze_columns_with_llm
from app.analysis.fallback_mapper import analyze_columns_fallback


logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """컬럼 매핑이 LLM·규칙 폴백 모두 실패했을 때(라우터가 500 으로 매핑)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


async def resolve_column_mapping(
    client: AsyncOpenAI | None,
    task_type: TaskType,
    columns: list[str],
    df,
) -> AnalysisResponse:
    """매핑 전략 결정: 무키→규칙폴백 / LLM→예외 시 규칙폴백 강등 / 폴백도 실패→AnalysisError."""

    def _rule_fallback(reason: str) -> AnalysisResponse:
        # 규칙 폴백도 extract_metadata 를 공유하므로 실패 가능 → 도메인 예외로(라우터가 500).
        try:
            return analyze_columns_fallback(task_type=task_type, columns=columns, df=df)
        except Exception as e:
            raise AnalysisError(
                f"컬럼 매핑 실패({reason}) 및 규칙 기반 폴백도 실패했습니다: {e}"
            )

    # 무키(client is None): 규칙 폴백
    if not client:
        logger.warning("OPENAI_API_KEY 미설정 → 규칙 기반 폴백 컬럼 매핑으로 강등합니다.")
        return _rule_fallback("no_key")

    # LLM 호출 실패(타임아웃/레이트리밋/파싱 등)도 500 대신 규칙 폴백으로 graceful degrade(D5c).
    try:
        return await analyze_columns_with_llm(
            client=client, task_type=task_type, columns=columns, df=df,
        )
    except Exception as e:
        # 조용한 품질 강등 지점 — 사후에 무엇이 왜 폴백됐는지 알 수 있어야 한다(G-06).
        logger.warning("LLM 컬럼 매핑 실패 → 규칙 기반 폴백으로 강등합니다: %r", e)
        return _rule_fallback("llm_error")
