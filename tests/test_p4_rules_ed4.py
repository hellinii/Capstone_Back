"""tests/test_p4_rules_ed4.py — 판 4 규칙 문장(판 4 라운드 S2 ⑦, 계획 부록 B20·B22·B24·A4·A11, 대응표 23·24·26·30).

- 부록 B 글자는 계획 표 그대로(아래 B20_*·B22_*·B24_* 는 계획 부록 B 의 승인 글을 옮겨 적은 것 — {…} 만 끼운다). 조사는 두
  저장소 공통 대조 표(tests/contract/number_wording_ed4.json 의 josa_cases)로 단정한다.
- 판 4 프론트는 고정 문구 18 을 보내지 않는다 — 판 4 5.5 규칙 문장은 18 이 있든 없든 B22 이고, 옛 요청 갈래('아슬아슬 합격')로
  빠지지 않는다(계획 9절 함정). 같은 요청의 판 3 쌍둥이는 그 갈래로 간다(갈래가 정말 갈림을 보인다).
- 폴백의 데이터/모델 가름은 저장값(칸 이름 9.1/9.2) 그대로, 판 4 권고표는 '6절 권고표'(A4), 판 4 판정 문장은 통과율 없이 충족 수만(A11).
- 판 4 의 모든 규칙 문장·결론 문장은 판 4 숫자 검문을 통과한다(자기 검문). 판 3 규칙 문장은 판 4 검문에 걸린다(이 검문이 빈
  검문이 아님).

판 3 규칙 문장이 한 글자도 바뀌지 않은 것은 얼린 정답(tests/golden_frozen/edition3/*.templates.json)이 본다.
"""
from __future__ import annotations

import copy
import json

import pytest

from app.narrative import fallback, generator
from app.narrative.fallback import _PROBLEM_TEMPLATES, problem_recommendation
from app.narrative.grounding import verify_grounding
from app.narrative.jobs import parse_request
from app.narrative.required_items import select_items
from app.narrative.schemas import NarrativeRequest, ProblemFact
from tests.p4_example_ed4 import CONTRACT, EXAMPLE_B20_FN, as_edition, edition4_cases, example_raw

TABLE = json.loads((CONTRACT / "number_wording_ed4.json").read_text(encoding="utf-8"))
CASES = edition4_cases()

# ── 계획 부록 B 승인 글(옮겨 적음) ─────────────────────────────────────────────────────
# B20 — 5.1 규칙 문장(이진, 상승폭 0.10 이상, 위험 방향 FN·FP). '(목표가 있으면)' 뒤 문장은 목표가 있을 때만.
B20_HEAD = "의뢰자가 더 위험하다고 밝힌 {side} 쪽 지표인 {name}{josa} F1이 가장 높은 결정 임계값 {t}에서 {v}이다."
B20_GOAL = "이 값은 목표 {target}{josa} {verb}."
B20_SIDE = {"fn": "거짓 음성(FN)", "fp": "거짓 양성(FP)"}
B20_VERB = {True: "충족한다", False: "충족하지 못한다"}
# B22 — 5.5 규칙 문장(판정 여유 작음) 두 문장. 없는 갈래는 뺀다.
B22_CI = "95% 신뢰구간이 목표값을 포함하는 {names}{josa} 판정의 여유가 작아, 표본이 달라지면 판정이 뒤집힐 수 있다."
B22_NO_CI = "신뢰구간이 없는 {names}{josa} 목표값과의 차이가 목표값의 3% 미만이라 판정의 여유가 작다."
# B24 — 판 4 근소 합격 폴백 권고: "목표값에 근접해 합격한" → "판정 여유가 작은 채 합격한"(나머지 글 그대로).
B24_FROM, B24_TO = "목표값에 근접해 합격한", "판정 여유가 작은 채 합격한"
# A4 — "9절 권고표" → "6절 권고표".
A4_TABLE = ("9절 권고표", "6절 권고표")


def _josa(word: str, pair: str) -> str:
    """공통 대조 표의 조사(표에 그 낱말·짝이 있어야 한다 — 없으면 시험이 표를 넓히라고 알린다)."""
    hits = [c["expected"] for c in TABLE["josa_cases"] if c["word"] == word and c["pair"] == pair]
    assert hits, f"공통 대조 표에 없음: {word} {pair}"
    return hits[0]


def _req(raw: dict) -> NarrativeRequest:
    req, _ = parse_request(copy.deepcopy(raw))
    return req


async def _templates(raw: dict) -> dict:
    result = await generator.generate_sections(None, _req(raw))
    return result.response.model_dump(mode="json")


def _texts(parts: dict) -> list[str]:
    out = list(parts["interpretation"].values()) + [parts["conclusion"][k] for k in ("benchmark", "narrative", "risks")]
    out += list(parts["recommendation_narrative"].values())
    out += [r[k] for r in parts["recommendations"] for k in ("category", "action", "expected_impact")]
    return [t for t in out if isinstance(t, str)]


def _conclusions(raw: dict):
    req = _req(raw)
    return {c.item_id: c for c in generator._prepare(req)[3]}


# ── 자기 검문 ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, raw", CASES, ids=[c[0] for c in CASES])
async def test_every_edition4_rule_sentence_passes_the_edition4_number_check(name, raw):
    req = _req(raw)
    _, _, wl, conclusions, _, _, _ = generator._prepare(req)
    texts = _texts(await _templates(raw))
    texts += [c.text for c in conclusions if c.author in ("ai", "ai_fact") and not c.open_ended]
    texts += [c.template_text for c in conclusions if c.template_text]
    got = verify_grounding(texts, wl, edition=4)
    assert got.checked > 0 and got.violations == [], (name, got.violations)
    joined = " ".join(texts)
    for p in req.fact_sheet.problems:  # 문제 이름은 프론트가 정한 글이다(판 3 입력은 판 3 이름 '아슬아슬 합격')
        joined = joined.replace(p.title, "□")
    for banned in ("통과율", "아슬아슬", "9절 권고표", "우수", "미흡"):
        assert banned not in joined, (name, banned)


async def test_edition3_rule_sentences_fail_the_edition4_number_check():
    """판 3 규칙 문장(넷째 자리 표기)은 판 4 허용 목록에 걸린다 — 위 자기 검문이 무엇이든 받는 검문이 아니다."""
    raw = example_raw()
    wl4 = generator._prepare(_req(raw))[2]
    texts3 = _texts(await _templates(as_edition(raw, 3)))
    got = verify_grounding(texts3, wl4, edition=4)
    assert "0.9767" in got.violations


# ── B22 — 판정 여유 작음(5.5) ─────────────────────────────────────────────────────────


def _risks(raw: dict) -> str:
    req = _req(raw)
    derived, refs, *_ = generator._prepare(req)
    return fallback.section_template(
        "8.3", req.fact_sheet, refs, derived, task_type="binary", client_inputs=req.client_inputs, edition=req.report_edition,
    )["conclusion.risks"]


def _metric(raw: dict, metric_id: str) -> dict:
    return next(m for m in raw["fact_sheet"]["metrics"] if m["metric_id"] == metric_id)


def test_b22_interval_sentence_for_metrics_whose_interval_contains_the_target():
    risks = _risks(example_raw())
    names = "Accuracy, Recall, F1 Score"
    assert B22_CI.format(names=names, josa=_josa(names, "은/는")) in risks
    assert "신뢰구간이 없는" not in risks


def test_b22_no_interval_sentence_and_both_sentences_in_order():
    raw = example_raw()
    m19 = _metric(raw, "M19")
    m19.update(near_target=True, ci_low=None, ci_high=None, ci_low_text=None, ci_high_text=None, ci_contains_target=None)
    risks = _risks(raw)
    first = B22_CI.format(names="Accuracy, Recall, F1 Score", josa=_josa("Accuracy, Recall, F1 Score", "은/는"))
    second = B22_NO_CI.format(names="Log Loss", josa=_josa("Log Loss", "은/는"))
    assert f"{first} {second}" in risks
    # 구간이 있는 쪽이 없으면 그 갈래는 뺀다.
    for mid in ("M1", "M3", "M4"):
        _metric(raw, mid).update(near_target=False)
    risks = _risks(raw)
    assert second in risks and "95% 신뢰구간이 목표값을 포함하는" not in risks


def test_b22_absent_when_no_metric_has_a_small_margin():
    raw = example_raw()
    for m in raw["fact_sheet"]["metrics"]:
        m["near_target"] = False
    assert "판정의 여유가 작" not in _risks(raw)


def test_edition4_never_falls_into_the_old_request_branch_without_sentence_18():
    """판 4 프론트는 18 을 보내지 않는다. 판 4 는 B22, 같은 요청의 판 3 쌍둥이(18 없음)는 옛 요청 갈래('아슬아슬 합격')."""
    raw = example_raw()
    assert not any(s["id"] == "18" for s in raw["fact_sheet"]["facts"]["fixed_sentences"])
    assert "아슬아슬" not in _risks(raw) and "판정의 여유가 작아" in _risks(raw)
    assert "아슬아슬 합격" in _risks(as_edition(raw, 3))
    # 18 이 실려 와도(판 3 프론트 꼴) 판 4 는 B22 — 18 을 보지 않는다.
    with18 = copy.deepcopy(raw)
    with18["fact_sheet"]["facts"]["fixed_sentences"].append(
        {"id": "18", "section": "8.3", "text": "목표값과의 차이가 목표값의 3% 미만인 지표: Recall 0.984(목표 0.97 이상, 충족)."}
    )
    assert _risks(with18) == _risks(raw)
    assert "목표값에 근소한 차이로" in _risks(as_edition(with18, 3))


# ── B20 — 위험 쪽 지표(5.1, 상승폭 0.10 이상) ─────────────────────────────────────────


def _tuning(raw: dict) -> dict:
    """예시의 상승폭을 0.10 이상(엔진 notable)으로 바꾼 사본 — 8.2-06(임계값 조정 여지)이 결론에 들어온다."""
    out = copy.deepcopy(raw)
    g = out["fact_sheet"]["f1_threshold_gain"]
    g.update(gain=0.12, notable=True, gain_text="0.120", max_f1=0.9844, current_f1=0.8644, current_f1_text="0.864")
    return out


def _gain_conclusion(raw: dict) -> str:
    return _conclusions(raw)["8.2-06"].text


def test_b20_follows_the_threshold_sentence_for_the_example():
    text = _gain_conclusion(_tuning(example_raw()))
    assert text.endswith(" " + EXAMPLE_B20_FN)
    g = _tuning(example_raw())["fact_sheet"]["f1_threshold_gain"]
    want = B20_HEAD.format(
        side=B20_SIDE["fn"], name="Recall", josa=_josa("Recall(M3)", "은/는"), t=g["best_threshold_text"], v=g["best_recall_text"],
    ) + " " + B20_GOAL.format(target="0.970", josa=_josa("0.970", "을/를"), verb=B20_VERB[True])
    assert want == EXAMPLE_B20_FN
    # 8.2 규칙 문장(템플릿)도 같은 결론 문장을 찍는다(작성 주체 ai).
    assert _conclusions(_tuning(example_raw()))["8.2-06"].template_text == text


def test_b20_not_meeting_and_false_positive_side():
    raw = _tuning(example_raw())
    rs = raw["fact_sheet"]["f1_threshold_gain"]["risk_side"]
    rs["meets_target_at_best"] = False
    assert _gain_conclusion(raw).endswith(
        B20_GOAL.format(target="0.970", josa=_josa("0.970", "을/를"), verb=B20_VERB[False])
    )
    # FP 쪽 — Precision(이 예시에서 목표 없는 참고 지표)이면 목표 문장이 없다.
    rs.update(side="fp", metric_id="M2", display_name="Precision", meets_target_now=None, meets_target_at_best=None)
    g = raw["fact_sheet"]["f1_threshold_gain"]
    josa = {"이": "은", "가": "는"}[_josa("Precision(M2)", "이/가")]  # 같은 끝소리 — 표의 '이/가' 줄로 '은/는'을 고른다
    head = B20_HEAD.format(side=B20_SIDE["fp"], name="Precision", josa=josa, t=g["best_threshold_text"], v=g["best_precision_text"])
    assert _gain_conclusion(raw).endswith(head)


def test_b20_absent_without_risk_side_and_without_tuning():
    raw = _tuning(example_raw())
    raw["fact_sheet"]["f1_threshold_gain"]["risk_side"] = None
    assert "의뢰자가 더 위험하다고 밝힌" not in _gain_conclusion(raw)
    assert "8.2-06" not in _conclusions(example_raw())  # 상승폭 0.10 미만 — 8.2-06 자체가 없다(5.6 은 지시문 유의가 맡는다)


async def test_b20_sentence_passes_the_edition4_number_check():
    raw = _tuning(example_raw())
    wl = generator._prepare(_req(raw))[2]
    parts = await _templates(raw)
    assert EXAMPLE_B20_FN in parts["conclusion"]["narrative"]
    assert verify_grounding([parts["conclusion"]["narrative"]], wl, edition=4).violations == []


# ── B24·A4 — 폴백 권고 ───────────────────────────────────────────────────────────────


def _problem(kind: str, section: str = "9.2") -> ProblemFact:
    return ProblemFact(id="P1", kind=kind, priority="MEDIUM", title="t", source_section="8.3", recommend_section=section,
                       evidence={}, note=None)


@pytest.mark.parametrize("kind", sorted(_PROBLEM_TEMPLATES))
def test_b24_only_core_near_target_changes_in_edition4(kind):
    r3 = problem_recommendation(_problem(kind), edition=3)
    r4 = problem_recommendation(_problem(kind), edition=4)
    if kind == "core_near_target":
        assert B24_FROM in r3.action and B24_TO not in r3.action
        assert r4.action == r3.action.replace(B24_FROM, B24_TO) and B24_TO in r4.action
        assert (r4.category, r4.expected_impact) == (r3.category, r3.expected_impact)
    else:
        assert r4 == r3


async def test_fallback_9_uses_edition4_table_name_and_splits_by_stored_section():
    parts = await _templates(example_raw())
    rec = parts["recommendation_narrative"]
    assert A4_TABLE[1] in rec["data_quality"] and A4_TABLE[1] in rec["model_ops"]
    assert A4_TABLE[0] not in json.dumps(parts, ensure_ascii=False)
    # 저장값은 칸 이름(9.1/9.2) — P2(같은 출처, 9.1)는 데이터 쪽, P1(근소 합격, 9.2)은 모델 쪽.
    probs = {p["id"]: p["recommend_section"] for p in example_raw()["fact_sheet"]["problems"]}
    assert probs == {"P1": "9.2", "P2": "9.1"}
    assert "P2(" in rec["data_quality"] and "P1(" not in rec["data_quality"]
    assert "P1(" in rec["model_ops"] and "P2(" not in rec["model_ops"]
    near = next(r for r in parts["recommendations"] if r["problem_id"] == "P1")
    assert B24_TO in near["action"]
    parts3 = await _templates(as_edition(_strip_same_source(example_raw()), 3))
    assert A4_TABLE[0] in parts3["recommendation_narrative"]["model_ops"]


def _strip_same_source(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    out["fact_sheet"]["problems"] = [p for p in out["fact_sheet"]["problems"] if p["kind"] != "same_source"]
    return out


# ── A11 — 통과율 없이 충족 수만 ───────────────────────────────────────────────────────


async def test_a11_verdict_sentence_drops_the_pass_rate():
    narrative4 = (await _templates(example_raw()))["conclusion"]["narrative"]
    narrative3 = (await _templates(as_edition(_strip_same_source(example_raw()), 3)))["conclusion"]["narrative"]
    assert "참고로 목표값이 설정된 지표 5개 중 5개가 목표를 충족하였다." in narrative4 and "통과율" not in narrative4
    assert "참고로 목표값이 설정된 지표 5개 중 5개가 목표를 충족하였다(통과율 100.0%)." in narrative3
    c4 = _conclusions(example_raw())["8.2-02"].text
    assert "통과율" not in c4 and "판정 대상 5개 중 5개가 목표를 충족했다" in c4


# ── D49 ② — 최약 클래스는 조건에 걸릴 때만 운영 위험(5.5) ────────────────────────────────


@pytest.mark.parametrize("low, far, want", [
    (False, False, "작은 표본(30건 미만 또는 전체의 5% 미만)이 아니고 매크로 F1 보다 0.1 이상 낮지도 않으므로, 이 클래스를 "
                   "언급하되 클래스 사이의 성능 차이를 운영 위험으로 쓰지 않는다."),
    (True, False, "작은 표본(30건 미만 또는 전체의 5% 미만)이어서 운영 위험으로 다룬다."),
    (False, True, "매크로 F1 보다 0.1 이상 낮아 운영 위험으로 다룬다."),
    (True, True, "작은 표본(30건 미만 또는 전체의 5% 미만)이고 매크로 F1 보다 0.1 이상 낮아 운영 위험으로 다룬다."),
])
def test_d49_2_weakest_class_is_an_operating_risk_only_under_the_d22_condition(low, far, want):
    raw = example_raw()
    raw["fact_sheet"]["facts"]["weakest_class"].update(low_support=low, far_below_macro=far)
    c = _conclusions(raw)["8.3-01"]
    assert want in c.text
    assert ("위험을 쓴다" in c.text) is (low or far)  # 용도상 위험을 쓰라는 말은 조건에 걸릴 때만
    assert c.template_text == "F1 이 가장 낮은 클래스는 F1 0.977인 malignant(악성 종양)이다."  # 템플릿은 사실 문장 그대로
    # 판 3 은 조건 없이 그대로(대응표 23 — 판 4 에서만).
    c3 = _conclusions(as_edition(_strip_same_source(raw), 3))["8.3-01"]
    assert "운영 위험" not in c3.text and "위험을 쓴다" in c3.text


# ── B18 — 5.4 칸이 빌 때 ──────────────────────────────────────────────────────────────


def test_b18_when_no_metric_is_in_the_reference_table():
    req = _req(example_raw())
    derived = generator._prepare(req)[0]
    conclusions = [c for c in select_items(req.fact_sheet, req.client_inputs, "binary", edition=4, derived=derived,
                                           benchmark_refs=[]) if c.item_id != "8.1-03"]
    kw = dict(task_type="binary", client_inputs=req.client_inputs, conclusions=conclusions)
    assert fallback.section_template("8.1", req.fact_sheet, [], derived, edition=4, **kw) == {"conclusion.benchmark": "해당 없음"}
    assert fallback.section_template("8.1", req.fact_sheet, [], derived, edition=3, **kw)["conclusion.benchmark"].startswith("비교 가능한")
    # 성적서가 고정 문구 16 을 찍으면(8.1-03 이 결론에 있음) 판 3 과 같이 빈 글·출처 fixed(작업 경로가 정한다).
    with16 = select_items(req.fact_sheet, req.client_inputs, "binary", edition=4, derived=derived, benchmark_refs=[])
    assert any(c.item_id == "8.1-03" for c in with16)
    got = fallback.section_template("8.1", req.fact_sheet, [], derived, edition=4, task_type="binary",
                                    client_inputs=req.client_inputs, conclusions=with16)
    assert got == {"conclusion.benchmark": ""}


# ── 판을 받는 새 인자는 기본값이 없다(빠뜨리면 바로 오류 — 계획 2.3) ─────────────────────


def test_new_edition_arguments_have_no_default():
    import inspect

    from app.narrative import required_items

    for fn in (problem_recommendation, required_items.verdict_sentence, required_items.error_ratio_sentence,
               fallback._recommendations, generator._align):
        param = inspect.signature(fn).parameters["edition"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty, fn.__name__
