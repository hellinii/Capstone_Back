"""
routers/narrative.py — LLM 서술 생성 API (성적서 7·8·9절).

- POST /api/generate-narrative         옛 경로(요청 하나로 서술, 서버 마감 120초). 운영 중인 옛 프론트가 쓴다.
- POST /api/narrative/jobs             작업 번호 경로 — 번호와 8.1 표를 즉시, 서술은 뒤에서(app.narrative.jobs)
- GET  /api/narrative/jobs/{job_id}    작업 단계·결과

평가(/api/evaluate)와 분리된 엔드포인트. 프론트가 평가 결과로 조립한 fact_sheet 를 받아
LLM(Claude) 또는 규칙 폴백으로 서술을 생성한다. Anthropic 키가 없거나 호출 실패/grounding 위반 시
규칙 기반 폴백으로 graceful degradation 하며, 평가 결과(KPI/차트)는 이 엔드포인트와 무관하게 유지된다.
"""
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.narrative import jobs
from app.narrative.schemas import NarrativeJobCreated, NarrativeJobOut, NarrativeRequest, NarrativeResponse
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


@router.post(
    "/narrative/jobs",
    response_model=NarrativeJobCreated,
    summary="서술 작업 만들기 (작업 번호 방식)",
    description=(
        "옛 경로와 같은 요청 본문(NarrativeRequest)을 받아 작업 번호와 8.1 벤치마크 표를 즉시 돌려준다. "
        "서술은 뒤에서 만들고 GET /api/narrative/jobs/{job_id} 로 단계와 결과를 본다. 사실 시트가 칸별 입력 "
        "상한을 넘으면 422 대신 상한 안으로 줄여 규칙 문장으로 완료한다. 작업 생성 시간당 상한을 넘으면 429."
    ),
    # 본문을 dict 로 받는 것은 입력 상한 초과를 422 대신 '규칙 문장으로 완료'로 바꾸기 위해서다(app.narrative.jobs).
    # 계약 문서에는 실제 모양(NarrativeRequest)을 싣는다.
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NarrativeRequest"}}},
        }
    },
)
async def create_narrative_job(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: dict[str, Any] = Body(...),
    session_factory=Depends(jobs.get_session_factory),
) -> NarrativeJobCreated:
    client = getattr(request.app.state, "anthropic_client", None)
    try:
        return await jobs.create_job(
            payload, client=client, session_factory=session_factory, background_tasks=background_tasks,
        )
    except ValidationError as exc:
        raise RequestValidationError(
            [{**e, "loc": ("body", *e["loc"])} for e in exc.errors(include_url=False, include_context=False)]
        ) from None
    except jobs.JobRateLimited:
        raise HTTPException(
            status_code=429, detail="서술 작업 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
        ) from None


@router.get(
    "/narrative/jobs/{job_id}",
    response_model=NarrativeJobOut,
    summary="서술 작업 조회",
    description=(
        "단계(queued·generating·checking·regenerating·done·failed)와, 끝났으면 서술 본문·칸별 출처를 돌려준다. "
        "멈춘 작업은 조회할 때 실패(lost)로 판정한다. 없는 번호는 404."
    ),
)
async def get_narrative_job(
    job_id: str = Path(max_length=64),
    session_factory=Depends(jobs.get_session_factory),
) -> NarrativeJobOut:
    job = await jobs.read_job(job_id, session_factory=session_factory)
    if job is None:
        raise HTTPException(status_code=404, detail="서술 작업을 찾을 수 없습니다.")
    return job
