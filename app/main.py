"""
main.py — FastAPI 앱의 진입점
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# VS Code ▶ 실행 버튼 클릭 시 (python app/main.py) 프로젝트 루트 경로 자동 등록
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # database.py 가 import 시점에 DATABASE_URL 을 읽으므로 그보다 먼저 실행해야 함
# ⚠️ 위 load_dotenv() 는 반드시 아래 app.core.database import 보다 앞에 있어야 한다.
#    (app.core.database 가 import 시점에 DATABASE_URL 을 읽음. isort/ruff 도입 시 이 순서가
#     깨지지 않도록 주의 — 필요 시 해당 블록에 `# isort: skip` 가드.)

from app.core.database import describe_backend, DATABASE_URL, init_db
from app.issuance.bootstrap import seed_organization
from app.analysis.router import router as analyze_router
from app.analysis.validation_router import router as validate_router
from app.evaluation.router import router as evaluate_router
from app.narrative.router import router as narrative_router
from app.issuance.router import router as reports_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 루트 로거를 한 번만 구성한다. 종전에는 print 만 있어 레벨도 필터링도 없었고
    # 조용한 강등이 아무 흔적을 남기지 않았다(G-06).
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    # 발급 메타 DB 준비: 테이블 생성(없으면) + 기관 시드(없으면). 설계 문서 §8.
    init_db()
    seed_organization()
    logger.info("발급 메타 DB 초기화 완료 (backend=%s)",
                "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY 미설정 — LLM 컬럼 자동 매핑 대신 규칙 기반 매핑이 동작하고 "
            "7·8·9절 서술도 폴백 문구로 대체됩니다."
        )
        app.state.openai_client = None
    else:
        # app.state에 클라이언트를 저장하여 모든 라우터에서 재사용 가능하게 함.
        # timeout 미설정 시 SDK 기본(~600s)으로 워커가 장시간 점유돼 자원 고갈 위험(D6a).
        app.state.openai_client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(45.0, connect=5.0),  # 전체 45s / 연결 5s
            max_retries=2,
        )
        logger.info("OpenAI 클라이언트 초기화 완료")
    yield
    logger.info("서버 종료")


app = FastAPI(
    title="ISO 4213 AI 분류 모델 평가 API",
    description="LLM 컬럼 매핑 및 ISO/IEC TS 4213:2022 기반 AI 분류 모델 평가지표 계산 서비스",
    version="0.3.0",
    lifespan=lifespan,
)

# 순수 JSON 본문 상한(G-03). 필드 단위 상한만으로는 부족하다 — 상한 위반을 판정하려면
# pydantic 이 본문을 **전부 파싱해야** 하고, 그 파싱이 이벤트 루프에서 일어난다.
# 실측: 18.8 MB JSON 이 422 로 거절되면서도 /health 를 3,179 ms 밀어냈다.
# 정상 서술 요청은 클래스 256개 상한에서도 1 MB 를 넘지 않는다.
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024


@app.middleware("http")
async def limit_json_body_size(request: Request, call_next):
    """선언된 Content-Length 가 상한을 넘는 JSON 요청을 **파싱 전에** 끊는다.

    멀티파트 업로드는 대상이 아니다 — 라우터의 공용 가드(G-04a)가 청크 단위로 읽으며
    이미 처리 전에 막는다. 여기서 또 재면 상한이 두 벌이 되고, 아래 '본문 비우기'가
    20 MB 파일에 대해서도 돌아 이득 없이 비용만 든다.

    응답 전에 본문을 읽어 버린다. 읽지 않고 닫으면 클라이언트가 전송 도중
    broken pipe 를 맞아 413 대신 네트워크 오류를 보게 된다(실측 확인). 비용이 컸던
    것은 읽기가 아니라 파싱·검증이므로, 읽고 버려도 방어 효과는 그대로다.

    Content-Length 가 없거나(chunked) 거짓이면 판단하지 않는다 — 그 경로는 스키마의
    필드 상한이 계속 막는다. 이것은 앞단 방어일 뿐이다.
    """
    declared = request.headers.get("content-length")
    content_type = request.headers.get("content-type", "")
    if (
        declared
        and declared.isdigit()
        and content_type.startswith("application/json")
        and int(declared) > MAX_JSON_BODY_BYTES
    ):
        logger.warning(
            "JSON 본문 크기 상한 초과로 거절 (path=%s, declared=%s, limit=%s)",
            request.url.path, declared, MAX_JSON_BODY_BYTES,
        )
        async for _ in request.stream():
            pass  # 파싱하지 않고 흘려보낸다
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"요청 본문이 너무 큽니다. JSON 은 최대 "
                          f"{MAX_JSON_BODY_BYTES // (1024 * 1024)} MB 입니다."
            },
        )
    return await call_next(request)


# ⚠️ CORS 는 **이 아래에서** 등록해야 한다. Starlette 은 나중에 추가된 미들웨어를 바깥에
#    두므로, 위 413 응답이 CORS 헤더를 달고 나가려면 CORS 가 가장 바깥이어야 한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["System"])
async def health_check(request: Request):
    """서버 상태 확인.

    DIAG=1 일 때만 진단 필드를 싣는다(G-07). 무인증 공개 엔드포인트이므로 평시에는
    응답이 종전과 동일하고, 배포 환경을 확인해야 할 때만 잠시 켠다. 이 진단은
    두 가지 관측을 curl 한 줄로 조달한다 —
      · 현재 프로덕션이 Postgres 로 붙어 있는지(REQUIRE_PERSISTENT_DB 를 켜기 전 필수)
      · 프록시 뒤에서 request.client.host 가 실제로 무엇인지(IP 기반 레이트리밋의 전제.
        레포 전체에 FORWARDED_ALLOW_IPS/--proxy-headers 가 0건이라, 모든 요청이 같은
        IP 로 보이면 IP 리밋이 전역 리밋으로 붕괴해 정상 사용자 1명이 전체를 막는다)
    """
    # 영속 여부는 **평시에도** 싣는다(ISSUES.md G-07). 종전에는 DIAG=1 을 켜야만 볼 수
    # 있었는데, 그것은 운영자가 특별한 절차를 밟아야 강등 사실을 안다는 뜻이었다.
    # 1차 라운드가 성적서 원본을 DB 에 넣은 뒤로 휘발성 DB 는 채번 중복만이 아니라
    # **발급된 성적서 자체를 잃는다.** 연결 문자열은 노출하지 않고 종류만 알린다.
    backend, persistent = describe_backend(DATABASE_URL)
    body = {"status": "ok", "db_backend": backend, "persistent": persistent}
    if os.getenv("DIAG") == "1":
        body["diagnostics"] = {
            "client_host": request.client.host if request.client else None,
            "forwarded_for": request.headers.get("x-forwarded-for"),
            "forwarded_proto": request.headers.get("x-forwarded-proto"),
        }
    return body

# ── 라우터(Router) 모듈 연결 ──
app.include_router(analyze_router)
app.include_router(evaluate_router)
app.include_router(validate_router)
app.include_router(narrative_router)
app.include_router(reports_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
