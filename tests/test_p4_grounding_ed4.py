"""tests/test_p4_grounding_ed4.py — 판 4 숫자 검문(판 4 라운드 S2 ⑥, 계획 2.7).

- 판 4 허용 목록은 판 4 사실 보기에 찍힌 표기 그대로다 — 셋째 자리와 판정 쪽으로 늘린 표기(0.9497·0.6166)만 받고, 판 3 처럼
  소수 1~4자리 반올림 변형을 넣지 않아 넷째 자리 표기(0.9767)는 위반이다. 통과율은 판 4 글에 쓰지 않아 넣지 않는다.
- 판 4 절 번호(1.1~6.2, '(4.3)', '5.1·5.2절', '(1.3 참조)', '부록 1.2')는 숫자로 보지 않는다. 판 3 절 번호 목록은 판 3 그대로다.
- 시험은 '판 3 이면 받는다/걸린다'를 함께 단정한다 — 우연히 없는 숫자로 통과하는 시험(바탕값 우연 허용)이 되지 않게.

판 3 허용 목록·검문 판정이 한 글자도 바뀌지 않은 것은 얼린 정답(tests/golden_frozen — edition3/*.fact_view.json 의 whitelist,
issued_checks_*.json)이 본다.
"""
from __future__ import annotations

import copy
import json

import pytest

from app.narrative import checks, fact_view, fact_view_ed4, generator, grounding
from app.narrative.grounding import _NUMBER_RE, _canon, build_number_whitelist, strip_context, verify_grounding
from app.narrative.schemas import NarrativeRequest
from app.narrative.sections import EDITION4_SECTION_NUMBERS, EDITION4_TITLES
from app.narrative.wording import num
from tests.p4_example_ed4 import (
    CONTRACT,
    EXAMPLE_ALLOWED,
    EXAMPLE_FOURTH_DIGIT,
    EXAMPLE_PASS_RATE,
    as_edition,
    edition4_cases,
    example_raw,
)

SLOTS = json.loads((CONTRACT / "narrative_slots_ed4.json").read_text(encoding="utf-8"))
CASES = edition4_cases()


def _prepared(raw: dict):
    req, _ = __import__("app.narrative.jobs", fromlist=["parse_request"]).parse_request(copy.deepcopy(raw))
    return req, generator._prepare(req)


def _wl(raw: dict) -> set:
    return _prepared(raw)[1][2]


def _canon_set(texts) -> set:
    return {_canon(t) for t in texts}


# ── 판 4 절 번호 ─────────────────────────────────────────────────────────────────────


def test_edition4_section_numbers_and_titles_match_the_contract():
    assert list(EDITION4_SECTION_NUMBERS) == SLOTS["edition4_section_numbers"]
    want = {s["slot"]: s["edition4_title"] for s in SLOTS["slots"]}
    want |= {s["slot"]: s["edition4_title"] for s in SLOTS["recommend_sections"]}
    assert EDITION4_TITLES == want and len(want) == 9


# 판 4 글에서 지우는 절 번호 꼴. 두째 값은 판 3 문맥 토큰으로는 숫자가 남는가(판 4 번호를 모른다 — 이 시험이 빈 갈래가
# 아님을 보인다). '6.2절' 처럼 '절'이 붙은 홑 번호는 판 3 의 일반 규칙도 지운다.
SECTION_TEXTS = [
    ("이 지표의 판정 여유는 작음이다(4.3).", True),
    ("판정 여유와 신뢰구간은 5.1·5.2절에서 다룬다.", True),
    ("이 위험에 대한 권고는 6.2절에서 다룬다.", False),
    ("제약은 앞에서 밝혔다(1.3 참조).", True),
    ("점검 결과는 부록 1.2에 있다.", True),
    ("자세한 해석은 5.1~5.6절을 본다.", True),
    ("판정 규칙(3.4)에 따라 합격이다.", True),
    ("5.5 운영 위험에서 다룬다.", True),
    ("(4.4, 4.5) 표를 함께 본다.", True),
    ("이 위험은 5.5 운영 위험에서 다시 다룬다.", True),
    ("권고는 6.1 데이터와 6.2 모델과 운영에 나눠 적었다.", True),
    ("관련 내용은 3.4절 5.1에서 다룬다.", True),  # '절 n.n' 꼴(S2·S3 검토 T12 — 이 줄이 빠지면 5.1 이 숫자로 걸린다)
]


@pytest.mark.parametrize("text, edition3_flags", SECTION_TEXTS)
def test_edition4_section_numbers_are_not_numbers(text, edition3_flags):
    got = verify_grounding([text], set(), edition=4)
    assert got.passed and got.checked == 0, (text, got)
    assert (not verify_grounding([text], set(), edition=3).passed) is edition3_flags, text


# 판 4 문맥 토큰을 넓혀도 놓치면 안 되는 진짜 숫자(실제로 숫자로 걸리는지 판 4 검문으로 단정).
MUST_CATCH = [
    ("Accuracy 는 (0.982)이다.", "0.982"),
    ("평균 지연은 4.3초이다.", "4.3"),
    ("Recall 은 5.5 이상이다.", "5.5"),
    ("이 값은 (5.5%)이다.", "5.5%"),
    ("판 3 자리는 (8.2)이다.", "8.2"),
    ("목표 대비 (6.3) 차이가 있다.", "6.3"),
    ("구간은 (4.3, 0.95)이다.", "0.95"),
    ("4.5배 많다.", "4.5"),
    ("부록 1.25 에 적었다.", "1.25"),
    ("값이 5.5 데이터 분포보다 크다.", "5.5"),  # 번호와 이름이 짝이 아니다(5.5 는 운영 위험, 데이터 분포는 5.3)
]


@pytest.mark.parametrize("text, token", MUST_CATCH)
def test_edition4_context_tokens_still_catch_real_numbers(text, token):
    got = verify_grounding([text], set(), edition=4)
    assert not got.passed and token in got.violations, (text, got)


def test_unknown_edition_is_refused():
    with pytest.raises(ValueError):
        grounding.context_tokens(5)
    with pytest.raises(ValueError):
        build_number_whitelist(NarrativeRequest.model_validate(example_raw()).fact_sheet, [], {}, edition=5)


# ── 판 4 허용 목록 ───────────────────────────────────────────────────────────────────


def test_example_allows_edition4_texts_and_rejects_fourth_digit():
    wl4 = _wl(example_raw())
    wl3 = _wl(as_edition(example_raw(), 3))
    assert wl4 != wl3
    for text in EXAMPLE_ALLOWED:
        assert verify_grounding([text], wl4, edition=4).passed, text
    assert EXAMPLE_FOURTH_DIGIT
    for text in EXAMPLE_FOURTH_DIGIT:
        assert not verify_grounding([text], wl4, edition=4).passed, text
        # 판 3 허용 목록에는 있다 — '우연히 없는 숫자'로 통과하는 시험이 아니다.
        assert verify_grounding([text], wl3, edition=3).passed, text
    # 통과율(판 4 는 충족 수만 — 부록 A11).
    assert _canon(EXAMPLE_PASS_RATE) in wl3 and _canon(EXAMPLE_PASS_RATE) not in wl4
    assert not verify_grounding([f"통과율은 {EXAMPLE_PASS_RATE}%이다."], wl4, edition=4).passed


def _view_value_tokens(view) -> list[str]:
    out: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out.append(_canon(repr(float(obj)) if isinstance(obj, float) else str(obj)))
        elif isinstance(obj, str):
            out.extend(_canon(t) for t in _NUMBER_RE.findall(strip_context(obj, edition=4)) if _canon(t) is not None)

    walk(view)
    return out


@pytest.mark.parametrize("name, raw", CASES, ids=[c[0] for c in CASES])
def test_every_number_in_the_edition4_view_is_allowed(name, raw):
    """판 4 사실 보기(지시문이 AI 에 보이는 것 — 덜어 내기 전)의 값 숫자는 모두 허용 목록 안(자기 검문)."""
    req, prepared = _prepared(raw)
    assert req.report_edition == 4
    view = fact_view.build_fact_view(
        req.fact_sheet, prepared[0], prepared[1], task_type=req.task_type.value, task_label="", report_purpose="",
        edition=4,
    )
    tokens = _view_value_tokens(view)
    assert tokens  # 빈 보기를 돌면 늘 통과한다
    assert sorted(set(tokens) - prepared[2]) == []


def _plain_values(fs) -> list[float]:
    """판정과 무관한 실수(판 4 는 셋째 자리로만 찍는다) — 클래스별 표·평균 줄·참고 지표 값."""
    values = [v for p in fs.per_class for v in (p.precision, p.recall, p.f1) if v is not None]
    values += [v for a in fs.class_averages for v in (a.precision, a.recall, a.f1) if v is not None]
    values += [m.value for m in fs.metrics if m.threshold is None and m.metric_id not in ("M21", "M22")]
    return [float(v) for v in values]


@pytest.mark.parametrize("name, raw", CASES, ids=[c[0] for c in CASES])
def test_fourth_digit_forms_of_plain_values_are_not_allowed(name, raw):
    """판정과 무관한 값의 넷째 자리 표기(판 3 wording.num)는 판 4 허용 목록에 없다 — 같은 값의 판 3 목록에는 있다(바탕값이 우연히
    그 숫자를 허용하지 않음을 보이는 짝 단정). 셋째 자리와 같은 넷째 자리(0.9500)와, 다른 사실이 같은 글자로 찍힌 값은 뺀다."""
    req, prepared = _prepared(raw)
    wl4 = prepared[2]
    wl3 = generator._prepare(NarrativeRequest.model_validate(as_edition(raw, 3)))[2]
    four = {num(v) for v in _plain_values(req.fact_sheet)}
    four = {t for t in four if "." in t and len(t.split(".")[1]) == 4}
    assert four, name  # 넷째 자리 값이 없는 사례를 돌면 시험이 빈다
    leaked = sorted(t for t in four if _canon(t) in wl4)
    # 늘린 표기·고객 입력 등 다른 사실이 같은 글자를 냈을 수 있다 — 그 경우는 이유가 사실 보기에 있어야 한다(아래에서 출처 확인).
    for t in leaked:
        view_text = json.dumps(generator._prepare(req)[4].view, ensure_ascii=False)
        client_text = json.dumps(req.client_inputs or {}, ensure_ascii=False)
        assert t in view_text or t in client_text, (name, t)
    assert all(_canon(t) in wl3 for t in four), name


def test_constants_only_when_their_judgement_is_on():
    """기준값은 판 3 과 같은 조건에서만(95% — 신뢰구간이 있을 때, 10%p — 분포 비교가 있을 때, 30·5% — 최약 클래스가 있을 때)."""
    raw = example_raw()
    assert {"95", "10", "30"} <= _wl(raw)
    # 95 는 구간이 있을 때만 — 판정 대상 지표가 있다고 판정 여유 칸 이름 때문에 받지 않는다(S2·S3 검토 D5).
    no_ci = copy.deepcopy(raw)
    for m in no_ci["fact_sheet"]["metrics"]:
        m.update(ci_low=None, ci_high=None, ci_low_text=None, ci_high_text=None, ci_contains_target=None)
    no_ci["fact_sheet"]["ci_method_note"] = None
    assert "95" not in _wl(no_ci)
    no_shift = copy.deepcopy(raw)
    no_shift["fact_sheet"]["facts"]["distribution_shift"] = []
    no_shift["fact_sheet"]["facts"]["distribution_shift_status"] = "not_provided"
    assert "10" not in _wl(no_shift)
    # 30 은 이 예시의 고객 입력('30% 층화 분할')에도 있다 — 고객 입력을 빼고 본다.
    weak = copy.deepcopy(raw)
    weak["client_inputs"] = None
    assert "30" in _wl(weak)
    no_weak = copy.deepcopy(weak)
    no_weak["fact_sheet"]["facts"]["weakest_class"] = None
    assert "30" not in _wl(no_weak)


def test_client_numbers_are_taken_as_written():
    """고객 입력의 숫자는 적힌 그대로만(판 4 는 반올림 변형을 넣지 않는다 — 고객 값이 지표 값과 같아 넷째 자리가 새는 구멍)."""
    raw = example_raw()
    raw["client_inputs"] = {**(raw.get("client_inputs") or {}), "note": "라벨 일치도 0.97674"}
    wl4 = _wl(raw)
    assert "0.97674" in wl4 and "0.9767" not in wl4
    raw3 = as_edition(raw, 3)
    assert "0.9767" in _wl(raw3)


# ── 배선: 판 4 요청은 칸 검사까지 판 4 로 ─────────────────────────────────────────────


def test_generator_builds_the_edition4_whitelist_for_edition4_requests():
    req = NarrativeRequest.model_validate(example_raw())
    derived, refs, wl, *_ = generator._prepare(req)
    assert wl == build_number_whitelist(req.fact_sheet, refs, derived, req.client_inputs, edition=4)
    assert wl != build_number_whitelist(req.fact_sheet, refs, derived, req.client_inputs, edition=3)


@pytest.mark.parametrize("section_id, field", [("8.3", "conclusion.risks"), ("7.3", "interpretation.additional_observations")])
def test_check_section_strips_edition4_section_numbers_only_for_edition4(section_id, field):
    req = NarrativeRequest.model_validate(example_raw())
    _, _, wl, conclusions, _, exempt, masks = generator._prepare(req)
    text = "판정 여유가 작은 지표는 4.3절 표와 같이 Recall 0.984이며(4.3), 권고는 6.2절에서 다룬다."
    kw = dict(whitelist=wl, verdict="PASS", conclusions=[], problem_ids=[], exempt=exempt, masks=masks)
    ok = checks.check_section(section_id, {field: text}, edition=4, **kw)
    assert ok.numbers == [] and "no_citation" not in ok.format, ok
    bad = checks.check_section(section_id, {field: text}, edition=3, **kw)
    assert "4.3" in bad.numbers  # 판 3 검문은 '(4.3)' 을 숫자로 본다 — 판을 넘기는 줄이 빠지면 판 4 글이 걸린다


def test_check_section_flags_fourth_digit_in_edition4_text():
    req = NarrativeRequest.model_validate(example_raw())
    _, _, wl, _, _, exempt, masks = generator._prepare(req)
    kw = dict(whitelist=wl, verdict="PASS", conclusions=[], problem_ids=[], exempt=exempt, masks=masks, edition=4)
    bad = checks.check_section("8.2", {"conclusion.narrative": "F1 Score 는 0.9767로 목표를 충족하였다."}, **kw)
    assert bad.numbers == ["0.9767"]
    good = checks.check_section("8.2", {"conclusion.narrative": "F1 Score 는 0.977로 목표를 충족하였다."}, **kw)
    assert good.numbers == []


# ── 칸 검사(check_section)가 하위 검사 셋(7.3 인용·기대 효과·D24)에 판을 넘기는가 (S2·S3 검토 T1) ─────────────────────
# 세 하위 함수(_cites_allowed_number·predicted_numbers·d24_present)는 기본값 없는 edition 을 받는다. 판 4 갈래에서 판을 빼먹으면
# '부록 1'·'부록 1.2'·'(4.4)' 같은 판 4 절 번호가 숫자로 남는다 — 아래는 판마다 결과가 갈리는 글로 세 줄을 하나씩 본다.


def _check_kw(raw: dict) -> tuple[dict, list]:
    req = NarrativeRequest.model_validate(raw)
    _, _, wl, conclusions, _, exempt, masks = generator._prepare(req)
    kw = dict(whitelist=wl, verdict="PASS", problem_ids=[], exempt=exempt, masks=masks)
    return kw, conclusions


def test_check_section_passes_edition_to_expected_impact_numbers():
    kw, _ = _check_kw(example_raw())
    rec = {"problem_id": "P1", "action": "평가 데이터를 따로 모은다.", "expected_impact": "부록 1.2 의 기준에 따라 재평가한다(4.4 참조)."}
    parts = {"recommendation_narrative.data_quality": "데이터 쪽 권고.", "recommendation_narrative.model_ops": "모델 쪽 권고.",
             "recommendations": [rec]}
    ok = checks.check_section("9", parts, conclusions=[], edition=4, **kw)
    assert ok.impact_numbers == [], ok
    # 판 3 은 판 4 절 번호를 모른다 — 같은 글이 예측 수치로 걸린다(판을 넘기는 줄이 빠지면 판 4 권고가 폴백으로 떨어진다).
    bad = checks.check_section("9", parts, conclusions=[], edition=3, **kw)
    assert "1.2" in bad.impact_numbers, bad


def test_check_section_passes_edition_to_the_7_3_citation_check():
    kw, _ = _check_kw(example_raw())
    assert "1" in kw["whitelist"]  # 이 예시의 거짓 음성 1건 — '부록 1' 의 1 이 허용 숫자와 겹친다
    field = "interpretation.additional_observations"
    # 판 4 절 번호와 허용 숫자 하나 — 인용이 있다.
    cited = checks.check_section("7.3", {field: "임계값 표(4.4)에서 Recall 은 0.984이다(부록 1.2 참조)."}, conclusions=[], edition=4, **kw)
    assert "no_citation" not in cited.format and cited.numbers == [], cited
    # 판 4 절 번호뿐 — 인용이 없다. 판 3 으로 보면 '부록 1' 의 1 을 허용 숫자로 읽어 인용이 있다고 잘못 본다.
    text = "자세한 점검 결과는 부록 1에서 다룬다."
    assert "no_citation" in checks.check_section("7.3", {field: text}, conclusions=[], edition=4, **kw).format
    assert "no_citation" not in checks.check_section("7.3", {field: text}, conclusions=[], edition=3, **kw).format


def test_check_section_passes_edition_to_the_d24_check():
    kw, conclusions = _check_kw(example_raw())
    d24 = [c for c in conclusions if c.item_id == "7.1-03"]
    assert d24 and d24[0].d24_values == (1.0, 0.333333333)
    field = "interpretation.confusion_analysis"
    # 선언 숫자(1건)가 판 4 절 번호와 붙어 있어도 있다.
    present = checks.check_section("7.1", {field: "놓친 오류는 1건이다(4.4). 그 뜻은 5.5절에서 다룬다."}, conclusions=d24, edition=4, **kw)
    assert present.d24_missing is False, present
    # 판 4 절 번호 속 숫자뿐 — 선언 숫자는 없다. 판 3 으로 보면 '부록 1' 의 1 을 선언 숫자로 읽는다.
    text = "의뢰자가 밝힌 위험 방향은 부록 1에 정리하였다."
    assert checks.check_section("7.1", {field: text}, conclusions=d24, edition=4, **kw).d24_missing is True
    assert checks.check_section("7.1", {field: text}, conclusions=d24, edition=3, **kw).d24_missing is False


@pytest.mark.parametrize("fn, args", [
    (checks.predicted_numbers, ("부록 1.2",)),
    (checks.d24_present, ("부록 1", (1.0,))),
    (verify_grounding, (["부록 1.2"], set())),
])
def test_inner_checks_have_no_default_edition(fn, args):
    """기본값 3 은 요청 경계(NarrativeRequest)와 legacy 입구뿐 — 안쪽 함수는 판을 받아야 한다."""
    with pytest.raises(TypeError):
        fn(*args)


# ── 판정 여유 칸 이름 속 숫자는 허용 목록 원천이 아니다(S2·S3 검토 D5) ──────────────────────────────────────────────


def _no_ci(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    for m in out["fact_sheet"]["metrics"]:
        m.update(ci_low=None, ci_high=None, ci_low_text=None, ci_high_text=None, ci_contains_target=None)
        m.update(near_target=bool(m["threshold"] is not None and m["margin"] is not None
                                  and abs(m["margin"]) < abs(m["threshold"]) * 0.03))
    out["fact_sheet"]["ci_method_note"] = None
    out["fact_sheet"]["facts"] = None
    return out


def test_made_up_3_and_95_are_caught_when_their_branch_is_absent():
    """판정 대상 지표가 있어도, 이 성적서에 '목표값의 3%' 갈래(구간 없는 판정 대상)가 없으면 지어낸 '3건' 은 위반이고, 구간이
    하나도 없으면 지어낸 '95%' 는 위반이다. 칸 이름(fact_view_ed4.MARGIN_KEY)에는 숫자가 없고, 뜻은 해당 갈래만 숫자와 함께 싣는다."""
    assert not any(ch.isdigit() for ch in fact_view_ed4.MARGIN_KEY)
    raw = dict(CASES)["ed3in_multilabel"]  # 판정 대상 둘 모두 구간이 있고, 데이터에 '3' 이 없는 사례
    req, prepared = _prepared(raw)
    assert any(m.threshold is not None for m in req.fact_sheet.metrics)
    wl = prepared[2]
    assert not verify_grounding(["경계 오류는 3건이다."], wl, edition=4).passed
    assert verify_grounding(["95% 신뢰구간이 목표값을 포함한다."], wl, edition=4).passed
    assert prepared[4].view["판정"]["판정 여유가 작음의 뜻"] == "95% 신뢰구간이 목표값을 포함하면 작음"
    # 구간을 모두 지운 사본 — 3% 갈래만 있다: '3%' 는 받고 '95%' 는 위반.
    no_ci = _no_ci(raw)
    req2, prepared2 = _prepared(no_ci)
    wl2 = prepared2[2]
    assert prepared2[4].view["판정"]["판정 여유가 작음의 뜻"] == "신뢰구간이 없는 지표는 목표값과의 차이가 목표값의 3% 미만이면 작음"
    assert verify_grounding(["목표값과의 차이가 목표값의 3% 미만이다."], wl2, edition=4).passed
    assert not verify_grounding(["95% 신뢰구간이 목표값을 포함한다."], wl2, edition=4).passed
