"""app/judge/config.py — 심판 모델·추론 강도·단가표.

코드 기본값 = 운영값. 환경변수는 호출마다 읽는다(배포 뒤 모델만 바꿀 때 코드를 고치지 않게).
모델 이름·추론 강도는 기준 묶음(criteria/model.json)에도 적혀 기준 지문에 들어간다 — 여기 기본값과 같은지는 시험이 본다.

상호작용
- 의존(import): os, re
- 사용처: app.judge.llm, scripts/judge_live.py
"""
import os
import re

# 심판 설계 4절 "모델 설정": 심판 gpt-6.1-sol(날짜 붙은 고정 버전 없음), 추론 강도 medium(4모듈 모두 같음).
# 공용 전처리(주장 추출)는 GPT-6 Luna 로 시작한다.
DEFAULT_JUDGE_MODEL = "gpt-6.1-sol"
DEFAULT_EXTRACT_MODEL = "gpt-6-luna"
DEFAULT_EFFORT = "medium"
# 공용 전처리(주장 추출)는 판단하지 않고 구절을 옮기는 일이라 추론 강도를 낮게 둔다(설계 4절의 '같은 강도'는 심판 4모듈 —
# J4 파일럿: medium 이면 추출 한 번에 60~70초, 출력의 절반 이상이 추론 토큰).
DEFAULT_EXTRACT_EFFORT = "low"
EFFORT_LEVELS = ("low", "medium", "high")

# 100만 토큰당 달러(입력, 캐시 입력, 출력). OpenAI 공식 가격표(2026-10-28 사용자 확인). 없는 모델은 비용 None(추정하지 않는다).
PRICES_PER_MTOK: dict[str, tuple[float, float, float]] = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
}

# 출력 상한(추론 포함). 채점 1회 출력 가정 1만 토큰(설계 5.5)의 넉넉한 위.
MAX_OUTPUT_TOKENS = 32000


def judge_model() -> str:
    """심판 4모듈 모델(`JUDGE_MODEL`, 비었으면 코드 기본값)."""
    return os.getenv("JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL


def extract_model() -> str:
    """공용 전처리 모델(`JUDGE_EXTRACT_MODEL`, 비었으면 코드 기본값)."""
    return os.getenv("JUDGE_EXTRACT_MODEL", "").strip() or DEFAULT_EXTRACT_MODEL


def judge_effort() -> str:
    """추론 강도(`JUDGE_EFFORT`). 비었거나 모르는 값이면 코드 기본값."""
    value = os.getenv("JUDGE_EFFORT", "").strip().lower()
    return value if value in EFFORT_LEVELS else DEFAULT_EFFORT


def extract_effort() -> str:
    """전처리 추론 강도(`JUDGE_EXTRACT_EFFORT`). 비었거나 모르는 값이면 코드 기본값."""
    value = os.getenv("JUDGE_EXTRACT_EFFORT", "").strip().lower()
    return value if value in EFFORT_LEVELS else DEFAULT_EXTRACT_EFFORT


def price_of(model: str | None) -> tuple[float, float, float] | None:
    """응답 모델 이름의 단가. 날짜가 붙은 응답 이름(gpt-6.1-sol-2026-…)은 앞 이름으로 찾는다. 모르면 None."""
    if not model:
        return None
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    m = _DATED.fullmatch(model)
    return PRICES_PER_MTOK.get(m.group(1)) if m else None


# 날짜 붙은 스냅샷 이름만 앞 이름의 단가로 본다(gpt-6.1-sol-mini 같은 다른 모델을 같은 값으로 세지 않게).
_DATED = re.compile(r"(.+)-\d{4}-\d{2}-\d{2}")
