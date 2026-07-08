"""
main.py — FastAPI 앱의 진입점
"""

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # database.py 가 import 시점에 DATABASE_URL 을 읽으므로 그보다 먼저 실행해야 함
# ⚠️ 위 load_dotenv() 는 반드시 아래 app.core.database import 보다 앞에 있어야 한다.
#    (app.core.database 가 import 시점에 DATABASE_URL 을 읽음. isort/ruff 도입 시 이 순서가
#     깨지지 않도록 주의 — 필요 시 해당 블록에 `# isort: skip` 가드.)

from app.core.database import DATABASE_URL, init_db, seed_organization
from app.analysis.router import router as analyze_router
from app.analysis.validation_router import router as validate_router
from app.evaluation.router import router as evaluate_router
from app.narrative.router import router as narrative_router
from app.issuance.router import router as reports_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 발급 메타 DB 준비: 테이블 생성(없으면) + 기관 시드(없으면). 설계 문서 §8.
    init_db()
    seed_organization()
    print(f"✅ 발급 메타 DB 초기화 완료 (backend={'sqlite' if DATABASE_URL.startswith('sqlite') else 'postgresql'})")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "LLM 기반 컬럼 자동 매핑 대신 룰 기반(Rule-based) 매핑이 작동하며, "
            "7, 8, 9절의 정성적 분석 서술 기능도 Fallback 문구로 대체됩니다."
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
        print("✅ OpenAI 클라이언트 초기화 완료")
    yield
    print("🛑 서버 종료")


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
