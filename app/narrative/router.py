"""
routers/narrative.py — LLM 서술 생성 API (성적서 7·8·9절).

평가(/api/evaluate)와 분리된 엔드포인트. 프론트가 평가 결과로 조립한 fact_sheet 를 받아
LLM(Claude) 또는 규칙 폴백으로 서술을 생성한다. Anthropic 키가 없거나 호출 실패/grounding 위반 시
규칙 기반 폴백으로 graceful degradation 하며, 평가 결과(KPI/차트)는 이 엔드포인트와 무관하게 유지된다.
"""
from fastapi import APIRouter, Request

from app.narrative.schemas import NarrativeRequest, NarrativeResponse
from app.narrative.service import generate_narrative

router = APIRouter(prefix="/api", tags=["Narrative"])


@router.post(
    "/generate-narrative",
    response_model=NarrativeResponse,
    summary="LLM 서술 생성 (7·8·9절)",
    description=(
        "평가 결과로 구성된 fact_sheet 를 받아 성적서 서술(정밀분석/종합소견/권고안)을 생성합니다. "
        "모든 수치는 fact_sheet 에서만 인용하며, 출력 숫자는 grounding 검증을 거칩니다. "
        "키 없음/호출 실패/검증 실패 시 규칙 기반 폴백으로 대체됩니다."
    ),
)
async def generate_narrative_endpoint(
    payload: NarrativeRequest,
    request: Request,
) -> NarrativeResponse:
    # app.state 의 Anthropic 클라이언트 재사용(없으면 None → 규칙 폴백). 컬럼 매핑의 OpenAI 클라이언트와 별개다.
    client = getattr(request.app.state, "anthropic_client", None)
    return await generate_narrative(client, payload)
