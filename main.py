"""
main.py — FastAPI 앱의 진입점
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

from routers import analyze, evaluate, validate

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "⚠️  경고: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "컬럼 자동 매핑(/api/analyze-columns) 기능은 작동하지 않으나, "
            "평가 실행(/api/evaluate) 및 매핑 확정(/api/confirm-mapping) 기능은 테스트하실 수 있습니다."
        )
        app.state.openai_client = None
    else:
        # app.state에 클라이언트를 저장하여 모든 라우터에서 재사용 가능하게 함
        app.state.openai_client = AsyncOpenAI(api_key=api_key)
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
app.include_router(analyze.router)
app.include_router(evaluate.router)
app.include_router(validate.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
