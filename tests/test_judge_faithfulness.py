"""tests/test_judge_faithfulness.py — 사실 일치(app/judge/faithfulness.py, 심판 라운드 J3, 계획 11-2). LLM 부분은 가짜로 배선만."""
from __future__ import annotations

import json
from types import SimpleNamespace as M

import pytest

from app.judge import faithfulness as fa
from app.judge.extract import Claim
from app.judge.reading import Piece
from app.judge.sources import Sources
from tests import judge_fakes as F

MARK = "ZQX고정표식"
FS = {
    "n_samples": 200,
    "metrics": [{"metric_id": "M1", "display_name": "Accuracy", "value": 0.505, "threshold": 0.9, "status": "fail", "is_core": True},
                {"metric_id": "M3", "display_name": "Recall", "value": 0.547, "threshold": 0.9, "status": "fail", "is_core": False}],
    "confusion_counts": [{"label": "1", "tp": 52, "fp": 56, "fn": 43, "tn": 49}],
    "facts": {"weakest_class": {"label": "0", "f1": 0.497}, "error_totals": {"misclassified": 99}},
}
METRICS = [M(**m) for m in FS["metrics"]]
SOURCES = Sources(FS, [], {"confusion": {"misclassified": 99}}, {"model": {"모델명": "ChurnAlert"}})
PIECES = [
    Piece("slot:7.1", "7.1", "5.2", "ai", "FN 은 43건으로 FP 56건보다 적다. 가장 약한 클래스는 0 이다. 경계 근처 표본이 많음을 시사한다."),
    Piece("slot:8.2", "8.2", "5.1", "ai", "Accuracy 0.505 는 목표에 못 미친다. 성능이 우수하다. 오류의 원인은 라벨 품질이다."),
    Piece("m:4.4-fixed17", "7.1", "4.4", "fixed", f"경계 오류는 12건이다. {MARK}"),
]


def claim(cid, kind, quote, piece="slot:7.1", field="", number_text="", polarity="", field_b=""):
    p = next(p for p in PIECES if p.id == piece)
    return Claim(cid, kind, quote, piece, p.slot, p.place, p.source, field, field_b, number_text, polarity, 0)


def answer(cid, **kw):
    base = {"claim_id": cid, "resolution": "resolved", "expr": "", "expr_b": "", "relation": "", "name_field": "", "named": "",
            "metric_id": "", "inference_basis": ""}
    return {**base, **kw}


async def run(claims, *replies):
    fake = F.FakeOpenAI(list(replies))
    return fake, await fa.judge_faithfulness(fake, PIECES, claims, SOURCES, METRICS, [])


async def test_numbers_matching_their_field_are_settled_by_code_without_llm():
    claims = [claim("C1", "number", "FN 은 43건", field="confusion_counts.1.fn", number_text="43"),
              claim("C2", "number", "Accuracy 0.505", piece="slot:8.2", field="metric.M1.value", number_text="0.505")]
    fake, res = await run(claims)
    assert fake.calls == []
    assert [v.verdict for v in res.verdicts] == ["match", "match"] and res.counts["checked"] == 2


async def test_number_not_matching_its_field_goes_to_agent_then_code_judges():
    # 추출이 붙인 칸과 어긋나면 바로 위반이 아니다 — 에이전트가 대상을 다시 찾는다(J4 파일럿 오탐)
    claims = [claim("C1", "number", "FP 56건", field="confusion_counts.1.fn", number_text="56"),  # 칸을 잘못 짚음
              claim("C2", "number", "FP 56건", field="confusion_counts.1.fp", number_text="55")]  # 진짜 틀린 숫자
    final = F.json_response({"claims": [answer("C1", expr="{confusion_counts.1.fp}"), answer("C2", expr="{confusion_counts.1.fp}")]})
    fake, res = await run(claims, final)
    assert len(fake.calls) == 1 and "C1" in fake.calls[0]["input"][0]["content"]
    assert [v.verdict for v in res.verdicts] == ["match", "violation"]
    assert res.violations()[0].slot == "7.1"


def test_name_matches_code_labels():
    assert fa.name_matches("FAIL", "종합 판정은 '최종 불합격'")
    assert fa.name_matches("cat", "가장 약한 클래스는 cat")
    assert not fa.name_matches("PASS", "종합 판정은 '최종 불합격'")


async def test_agent_tool_loop_then_code_judges_its_answer():
    claims = [
        claim("C1", "comparison", "FN 은 43건으로 FP 56건보다 적다"),
        claim("C2", "name", "가장 약한 클래스는 0 이다"),
        claim("C3", "number", "43건", number_text="43"),  # 칸을 못 정한 숫자 — 식으로
        claim("C4", "degree", "성능이 우수하다", piece="slot:8.2", polarity="positive"),
    ]
    final = F.json_response({"claims": [
        answer("C1", expr="{confusion_counts.1.fn}", expr_b="{confusion_counts.1.fp}", relation="lt"),
        answer("C2", name_field="facts.weakest_class.label", named="0"),
        answer("C3", expr="{facts.error_totals.misclassified} - {confusion_counts.1.fp}"),
        answer("C4", metric_id=""),
    ]})
    fake, res = await run(claims, F.response([F.reasoning_item(), F.function_call_item("lookup_value", {"field": "confusion_counts.1.fn"}, "call_1")]),
                          final)
    assert len(fake.calls) == 2 and res.tool_calls == 1
    # 도구 결과(값·출처)가 다음 턴에 되돌아갔다
    out = json.loads(fake.calls[1]["input"][-1]["output"])
    assert out == {"field": "confusion_counts.1.fn", "value": 43, "source": "성적서 데이터"}
    assert [v.verdict for v in res.verdicts] == ["match", "match", "match", "violation"]  # 핵심 지표 미달인데 '우수'
    assert fake.calls[0]["model"] == "gpt-6.1-sol" and [t["name"] for t in fake.calls[0]["tools"]] == [
        "lookup_value", "find_fields", "read_context", "calc"]


async def test_code_catches_wrong_relation_name_and_number():
    claims = [claim("C1", "comparison", "FN 은 43건으로 FP 56건보다 적다"), claim("C2", "name", "가장 약한 클래스는 0 이다"),
              claim("C3", "number", "43건", number_text="43")]
    final = F.json_response({"claims": [
        answer("C1", expr="{confusion_counts.1.fn}", expr_b="{confusion_counts.1.fp}", relation="gt"),
        answer("C2", name_field="facts.weakest_class.label", named="1"),
        answer("C3", expr="{confusion_counts.1.fp}"),
    ]})
    _, res = await run(claims, final)
    assert [v.verdict for v in res.verdicts] == ["violation", "violation", "violation"]


async def test_unresolved_missing_or_unknown_field_is_unverifiable_not_violation():
    claims = [claim("C1", "number", "43건", number_text="43"), claim("C2", "number", "43건", number_text="43"),
              claim("C3", "degree", "성능이 우수하다", piece="slot:8.2", polarity="positive")]
    final = F.json_response({"claims": [answer("C1", resolution="no_data"), answer("C2", expr="{no.such.field}"),
                                        answer("C3", resolution="target_unclear")]})
    _, res = await run(claims, final)
    assert [v.verdict for v in res.verdicts] == ["unverifiable", "unverifiable", "target_unclear"]
    assert res.counts["checked"] == 0


async def test_inference_marker_decided_by_code_and_client_input_count():
    claims = [claim("C1", "inference", "경계 근처 표본이 많음을 시사한다"),
              claim("C2", "inference", "오류의 원인은 라벨 품질이다", piece="slot:8.2")]
    final = F.json_response({"claims": [answer("C1", inference_basis="data"), answer("C2", inference_basis="client_input")]})
    _, res = await run(claims, final)
    assert [v.verdict for v in res.verdicts] == ["inference_ok", "inference_violation"]
    assert res.counts["client_input_assertive"] == 1 and len(res.violations()) == 1


async def test_agent_turn_cap_is_failure():
    loop = [F.response([F.function_call_item("find_fields", {"query": "fn"})]) for _ in range(12)]
    fake, res = await run([claim("C1", "number", "43건", number_text="43")], *loop)
    assert res.failure == "agent_turns" and len(fake.calls) == 12


async def test_llm_failure_propagates():
    _, res = await run([claim("C1", "number", "43건", number_text="43")], F.response([F.message_item("x", refusal=True)]))
    assert res.failure == "refusal" and res.verdicts == []


async def test_fixed_and_table_pieces_never_sent():
    fake, _ = await run([claim("C1", "number", "43건", number_text="43")], F.json_response({"claims": [answer("C1")]}))
    assert MARK not in json.dumps(fake.calls, ensure_ascii=False)


def test_tools_read_context_and_calc():
    out = fa._tool_output("read_context", {"piece_id": "slot:7.1"}, SOURCES, PIECES[:2])
    assert [p["id"] for p in out["pieces"]] == ["slot:7.1"]
    assert fa._tool_output("calc", {"expr": "43/99"}, SOURCES, PIECES)["value"] == pytest.approx(43 / 99)
    assert "error" in fa._tool_output("calc", {"expr": "__import__('os')"}, SOURCES, PIECES)
    assert fa._tool_output("lookup_value", {"field": "client.model.모델명"}, SOURCES, PIECES)["source"] == "고객 입력"


def test_eval_expr():
    assert fa.eval_expr("{confusion_counts.1.fn} / {facts.error_totals.misclassified}", SOURCES) == pytest.approx(43 / 99)
    assert fa.eval_expr("{nope} + 1", SOURCES) is None
    assert fa.eval_expr("", SOURCES) is None


async def test_claim_the_agent_did_not_answer_is_unverifiable():
    claims = [claim("C1", "number", "43건", number_text="43"), claim("C2", "name", "가장 약한 클래스는 0 이다")]
    _, res = await run(claims, F.json_response({"claims": [answer("C1", expr="{confusion_counts.1.fn}")]}))
    assert [v.verdict for v in res.verdicts] == ["match", "unverifiable"]


@pytest.mark.parametrize("sentence, marked", [
    ("도메인별로 상이할 수 있음", True), ("판정이 달라질 가능성은 낮다", True), ("성능 저하 위험이 존재한다", True),
    ("경계 근처 표본이 많음을 시사한다", True), ("오류의 원인은 라벨 품질이다", False),
])
def test_inference_marker_stems(sentence, marked):
    assert fa.inference_marked(sentence) is marked


def test_pilot2_rules():
    assert fa._relation_holds(0.48007, 0.48, "eq") is True and fa._relation_holds(0.4806, 0.48, "eq") is False
    assert fa.name_matches("해지할 가입자를 넘긴다.", "해지할 가입자를 넘긴다")
    assert SOURCES.lookup("rule.imbalance_ratio_threshold") == {"field": "rule.imbalance_ratio_threshold", "value": 1.5,
                                                                  "source": "판정 규칙 기준값"}


def test_client_text_partial_quote():
    assert fa.client_text_matches("2026년 1분기 해지 신청 기록에서 뽑은 200명", "2026년 1분기 해지 신청 기록")
    assert not fa.client_text_matches("2026년 1분기 해지 신청 기록에서 뽑은 200명", "2025년 기록")
    assert not fa.client_text_matches("긴 고객 입력 글", "글")  # 너무 짧은 조각은 맞다고 보지 않는다
