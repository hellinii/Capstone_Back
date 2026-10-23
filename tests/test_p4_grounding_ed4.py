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

from app.narrative import checks, fact_view, generator, grounding
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
    # 95·3 은 판정 대상 지표가 있으면 판정 여유 칸 이름(뜻 — 95% 신뢰구간·목표값의 3%)으로도 보인다. 구간도 판정 대상도 없으면 없다.
    no_ci = copy.deepcopy(raw)
    for m in no_ci["fact_sheet"]["metrics"]:
        m.update(ci_low=None, ci_high=None, ci_low_text=None, ci_high_text=None, ci_contains_target=None)
    no_ci["fact_sheet"]["ci_method_note"] = None
    assert "95" in _wl(no_ci)
    for m in no_ci["fact_sheet"]["metrics"]:
        m.update(threshold=None, status="info", margin=None, margin_text=None, value_text=None, near_target=False)
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
