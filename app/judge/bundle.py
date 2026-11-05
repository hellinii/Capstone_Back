"""app/judge/bundle.py — 고객용 심판 묶음(계획 14-2, 계약 tests/contract/judge_bundle_ed4.json).

판 4 서술 작업 결과의 `judge` 칸을 만든다. 실을 축은 여기 한 곳(PUBLISHED_AXES)이 정한다 — 검증(7.7)에서 축을 빼면 이 값만
바꾼다. 근거 인용·사유·주장 내용·재생성 횟수·토큰·비용·시간·기준 지문은 싣지 않는다(계약 never_in_bundle).

상호작용
- 의존(import): app.judge.config·criteria
- 사용처: app.narrative.jobs(판 4 갈래), tests
"""
from __future__ import annotations

from app.judge import config, criteria

# 필수 내용(coverage)은 뺐다 — V2 결함 검사에서 D03(권고 하나의 근거 문장 삭제)을 못 잡았다: 항목 '권고-1(각 권고의 근거)'을 항목
# 하나로 보아 다른 권고의 근거로 '서술됨'이라 했다. 검증 README 미리 정한 기준 9("필수 내용·표기 축만 미달이면 그 축만 빼고 켠다"),
# 민서 2026-11-01. 필수 내용 점검은 내부 기록으로만 계속 돌고(검증·분석용) 발급을 막지 않는다(core·safety 도 이 목록을 본다).
# 표기(readability)도 뺐다 — V2 사람 일치도 미달(시스템↔사람 0.84 vs 사람↔사람 1.00, 표기 축 0.58: 어긋남은 모두 AI '읽기 어려움'
# 표시). 같은 기준 9, 민서 2026-11-04. 표기 점검도 내부 기록으로만 돌고, 그 호출 실패로 보류하지 않는다(core).
# 사실 일치(faithfulness)도 뺐다 — 평가 단계(2026-11-05, 사전 등록·개발에 쓰지 않은 결과서 3건)에서 축 수준 미탐지(F03 필드
# 오지정·F08 문장 단위 추론 표시)와 판정자 오판 과잉 차단(오류 없는 원본·변형 30회 중 17회, 오판은 모두 사실 일치)이 났다. 사전
# 등록 해석 규칙 §8("그 축을 고객 부록에서 뺀다"), 민서. 사실 일치는 발급도 막지 않는다 — 숫자 위반 칸만 규칙 문장으로 바꾼다
# (aggregate.issuance·ladder). 기록 JUDGE_ROUND/평가/05_결과/평가_실험기록.md.
PUBLISHED_AXES = ("consistency",)


def _head(gate: str, held_kind: str | None, retry_allowed: bool) -> dict:
    return {"gate": gate, "held_kind": held_kind, "retry_allowed": retry_allowed,
            "criteria_version": criteria.version(), "model": config.judge_model(), "axes": list(PUBLISHED_AXES)}


def held(kind: str, *, retry_allowed: bool) -> dict:
    return {**_head("held", kind, retry_allowed), "counts": None}


def _axis_counts(axis: str, s: dict) -> dict:
    if axis == "faithfulness":
        return {"checked": s["checked"], "matched": s["matched"]}
    if axis == "consistency":
        return {"checked": s["checked"], "contradictions": s["contradictions"]}
    if axis == "coverage":
        return {"applicable": s["applicable"], "described": s["described"], "not_applicable": s["not_applicable"],
                "items": [{"id": i["id"], "described": i["described"], "place": i["place"]} for i in s["items"]]}
    return {"rules": s["rules"], "violations": s["violations"]}


def passed(scorecard: dict) -> dict:
    """싣는 축만 읽는다 — 뺀 축은 점수표에 없어도(내부 점검 실패) 묶음이 만들어진다."""
    return {**_head("passed", None, False), "counts": {a: _axis_counts(a, scorecard[a]) for a in PUBLISHED_AXES}}
