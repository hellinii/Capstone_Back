"""scripts/dev_fake_llm_server.py — 로컬 확인용 가짜 LLM 개발 서버(심판 라운드 J5b, 계획 14-7).

브라우저 스모크(J5b·J6·J7)에서 판 4 작업을 "검증 중 → 통과/보류"까지 돌려 보려고 쓴다. 실호출은 하지 않는다.

- 앱 시작이 끝난 뒤 앱의 모든 LLM 클라이언트(app.state 의 *_client — 서술 Anthropic, 매핑·심판 OpenAI)를 가짜로 바꾸고,
  첫 요청을 받기 전에 "모든 클라이언트가 가짜 클래스"인지 단정해 아니면 종료한다.
- 키는 지우지 않고 빈 문자열로 덮는다 — 지우면 app.main 의 load_dotenv() 가 .env 에서 다시 채운다.
- DB 는 임시 sqlite 만 받는다. 주소 검사는 import 뒤 실제 값(app.core.database.DATABASE_URL)으로 한다.
- 운영 진입점(app.main:app 을 uvicorn 으로)과 다르다. app/ 는 scripts/ 를 import 하지 않는다(시험이 단정).

심판 모드(--judge):
- pass  : 주장 하나(숫자 낱말)·조회 자료 없음(no_data) → 위반 0·모순 0 → 통과
- error : 심판 호출이 늘 연결 끊김 → 보류(judge_error, 일시적 — 다시 만들기 줌)
- refuse: 심판이 거절 → 보류(judge_error, 다시 만들기 없음)
매핑(OpenAI chat)은 늘 연결 끊김 — 앱이 규칙 매핑으로 강등한다.

쓰는 법: venv/bin/python scripts/dev_fake_llm_server.py --port 8000 --judge pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SECTION_TEXTS = {
    "section_7_1": "혼동행렬의 오분류 방향을 해석한 서술이다(개발용 가짜).",
    "section_7_2": "클래스 분포를 해석한 서술이다(개발용 가짜).",
    "section_7_3": "",
    "section_8_1": "내부 참조 범위와 비교한 서술이다(개발용 가짜).",
    "section_8_2": "종합 판정의 사유를 적은 서술이다(개발용 가짜).",
    "section_8_3": "운영상 위험을 적은 서술이다(개발용 가짜).",
    "section_9_1": "데이터 측면의 권고 서술이다(개발용 가짜).",
    "section_9_2": "모델 측면의 권고 서술이다(개발용 가짜).",
}
JUDGE_MODES = ("pass", "error", "refuse")


class FakeAnthropic:
    """서술 생성 — 요청한 칸 스키마의 필드만 채운 BetaMessage 를 돌려준다."""

    def __init__(self):
        self.beta = self
        self.messages = self
        self.calls = 0

    def with_options(self, **_kw):
        return self

    async def create(self, **kwargs):
        from anthropic.types.beta import BetaMessage

        self.calls += 1
        schema = kwargs["output_config"]["format"]["schema"]
        out = {}
        for name, spec in schema["properties"].items():
            if name == "recommendations":
                ids = spec["items"]["properties"]["problem_id"]["enum"]
                out[name] = [{"problem_id": i, "category": "점검", "action": "해당 문제를 점검하고 재평가한다.",
                              "expected_impact": "판정의 근거가 분명해진다."} for i in ids]
            else:
                out[name] = SECTION_TEXTS.get(name, "")
        usage = {"input_tokens": 1000, "output_tokens": 500,
                 "iterations": [{"type": "message", "model": "claude-sonnet-5-5", "input_tokens": 1000, "output_tokens": 500,
                                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]}
        return BetaMessage.model_validate({
            "id": "msg_dev_fake", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
            "stop_reason": "end_turn", "stop_details": None, "stop_sequence": None, "usage": usage,
        })


def _judge_answer(kwargs) -> dict:
    """심판 모듈(text.format.name)마다 '문제 없음' 답."""
    fmt = kwargs["text"]["format"]
    name, schema = fmt["name"], fmt["schema"]
    text = kwargs["input"][0]["content"] if isinstance(kwargs["input"], list) else kwargs["input"]
    if name == "claims":
        for block in str(text).split("\n\n"):
            head, _, body = block.partition("\n")
            tok = next((t for t in body.split() if any(ch.isdigit() for ch in t)), None)
            if head.startswith("[slot:") and tok:
                return {"claims": [{"kind": "number", "quote": tok, "piece_id": head[1:].split(" | ")[0], "field": "",
                                    "field_b": "", "number_text": "", "polarity": ""}]}
        return {"claims": []}
    if name == "faithfulness":
        return {"claims": [{"claim_id": "C1", "resolution": "no_data", "expr": "", "expr_b": "", "relation": "",
                            "name_field": "", "named": "", "metric_id": "", "inference_basis": ""}]}
    if name == "coverage":
        ids = schema["properties"]["items"]["items"]["properties"]["id"]["enum"]
        return {"items": [{"id": i, "present": False, "quote": "", "piece_id": ""} for i in ids]}
    if name == "readability":
        slots = schema["properties"]["slots"]["items"]["properties"]["slot"]["enum"]
        return {"slots": [{"slot": s, "present": False, "quote": "", "piece_id": ""} for s in slots]}
    if name == "consistency":
        return {"checks": [{"type": t, "present": False, "quote_a": "", "piece_a": "", "quote_b": "", "piece_b": ""}
                           for t in ("verdict_vs_recommendation", "weakness_vs_recommendation", "analysis_vs_finding")]}
    raise ValueError(f"모르는 심판 출력 형식: {name}")


class FakeOpenAI:
    """매핑(chat)은 늘 연결 끊김, 심판(responses)은 모드대로."""

    def __init__(self, mode: str):
        if mode not in JUDGE_MODES:
            raise ValueError(mode)
        self.mode = mode
        self.responses = _Responses(self)
        self.chat = self
        self.completions = self
        self.calls = 0

    def with_options(self, **_kw):
        return self

    async def create(self, **_kwargs):  # chat.completions.create
        import httpx
        import openai

        raise openai.APIConnectionError(request=httpx.Request("POST", "https://dev-fake.invalid/chat"))


class _Responses:
    def __init__(self, owner: FakeOpenAI):
        self.owner = owner

    async def create(self, **kwargs):
        import httpx
        import openai
        from openai.types.responses import Response

        self.owner.calls += 1
        if self.owner.mode == "error":
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://dev-fake.invalid/responses"))
        part = ({"type": "refusal", "refusal": "개발용 가짜 거절"} if self.owner.mode == "refuse"
                else {"type": "output_text", "text": json.dumps(_judge_answer(kwargs), ensure_ascii=False), "annotations": []})
        return Response.model_validate({
            "id": f"resp_dev_{self.owner.calls}", "object": "response", "created_at": 1, "model": kwargs["model"],
            "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"id": f"msg_dev_{self.owner.calls}", "type": "message", "role": "assistant", "status": "completed",
                        "content": [part]}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens": 100,
                      "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1100},
        })


FAKE_CLASSES = (FakeAnthropic, FakeOpenAI)


def assert_all_fake(state) -> None:
    """app.state 의 모든 *_client 가 가짜인지 — 아니면 SystemExit(진짜 클라이언트로 요청을 받지 않는다)."""
    names = [n for n in getattr(state, "_state", {}) if n.endswith("_client")]  # starlette State 는 _state dict 에 둔다
    bad = [n for n in names if not isinstance(getattr(state, n), FAKE_CLASSES)]
    if not names or bad:
        raise SystemExit(f"가짜가 아닌 LLM 클라이언트가 있다: {bad or '(클라이언트 없음)'} — 개발 서버를 멈춘다")


def temp_sqlite_only(url: str) -> None:
    tmp = os.path.realpath(tempfile.gettempdir())
    path = url[len("sqlite:///"):] if url.startswith("sqlite:///") else None
    if path is None or not os.path.realpath(path).startswith(tmp):
        raise SystemExit(f"개발 서버는 임시 sqlite 만 쓴다(지금 {url!r}) — 운영·로컬 DB 를 건드리지 않게 멈춘다")


def build_app(mode: str):
    """키를 빈 문자열로 덮고 임시 DB 를 정한 뒤 앱을 import 해 가짜 클라이언트를 끼운 앱을 돌려준다."""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""
    if not os.environ.get("DATABASE_URL", "").startswith("sqlite:///"):
        os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='dev_fake_llm_')}/dev.db"
    from app.core import database
    from app.main import app

    temp_sqlite_only(database.DATABASE_URL)
    original = app.router.lifespan_context

    @asynccontextmanager
    async def fake_lifespan(a):
        async with original(a) as state:
            a.state.anthropic_client = FakeAnthropic()
            a.state.openai_client = FakeOpenAI(mode)
            assert_all_fake(a.state)
            yield state

    app.router.lifespan_context = fake_lifespan
    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--judge", choices=JUDGE_MODES, default="pass")
    args = parser.parse_args(argv)
    app = build_app(args.judge)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
