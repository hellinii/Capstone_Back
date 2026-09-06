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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # database.py 가 import 시점에 DATABASE_URL 을 읽으므로 그보다 먼저 실행해야 함
# ⚠️ 위 load_dotenv() 는 반드시 아래 app.core.database import 보다 앞에 있어야 한다.
#    (app.core.database 가 import 시점에 DATABASE_URL 을 읽음. isort/ruff 도입 시 이 순서가
#     깨지지 않도록 주의 — 필요 시 해당 블록에 `# isort: skip` 가드.)

from app.core.database import DATABASE_URL, init_db
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["System"])
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok"}

# ── 라우터(Router) 모듈 연결 ──
app.include_router(analyze_router)
app.include_router(evaluate_router)
app.include_router(validate_router)
app.include_router(narrative_router)
app.include_router(reports_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
