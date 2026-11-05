"""app/judge — 판 4 서술 심판(LLM-as-judge 평가 시스템).

서술 작업이 판 4 서술을 다 만든 직후, 초안이 나오기 전에 서술을 검사한다. 4축(사실 일치·필수 내용·일관성·표기)을
따로 재고, 모순이 있으면 발급을 막는다(사실 위반은 막지 않고 숫자 위반 칸만 규칙 문장으로 바꾼다 — 2026-11-05, aggregate.issuance).
심판 사유는 서술 생성 지시문에 넣지 않는다.

import 규칙(tests/test_judge_import_rules.py 가 지킨다)
- 서술 생성 지시문 쪽 모듈(app.narrative 의 prompt·generator·checks·grounding·required_items·fact_view*·fallback·wording*)은
  app.judge 를 import 하지 않는다. 작업 흐름(app.narrative.jobs)만 심판을 부른다.
- app.judge 는 서술 쪽 판정 규칙(app.narrative.checks·grounding·required_items)을 import 하지 않는다 — 서술 검문이 바뀌면
  심판 기준이 몰래 바뀌기 때문이다. 필요한 작은 도구는 기준 묶음(criteria/)에 복사해 둔다.

모듈
- config: 모델 이름·추론 강도·단가표(모르면 None)
- llm: OpenAI Responses 호출 한 곳(재시도·시간·토큰·비용 기록)
- criteria/: 기준 고정 묶음(체크리스트·주장 규칙·프롬프트 등)과 지문(MANIFEST)
- criteria_lock: 검증 기록 잠금(14-10) — 지금 기준 지문이 검증 기록(validated.json, J11 이 씀)과 다르면 판 4 보류
"""
