"""tests/test_p4_s6_decisions_ed4.py — 판 4 라운드 S6 뒤 사용자 결정(2026-10-24)의 백엔드 쪽.

- 결정 4: 판 4 규칙 문장·폴백의 정수 건수는 1,000 이상이면 천 단위 쉼표(wording_ed4.ed4_count — 프론트 ed4Count 와 같은 대조 표 count 줄).
  소수·지연시간(D4)은 그대로. 숫자 허용 목록은 쉼표를 떼고 견주므로 '2,687' 을 받고, 지어낸 '2,688' 은 받지 않는다.
- 결정 5: 판 4 클래스 이름은 '이름(설명)', 설명에 괄호가 있으면 '이름[설명]'(class_name_ed4), 조사는 대괄호 앞 이름(wording_ed4.josa),
  이름 바로 뒤 건수는 이름이 숫자로 끝나면 '{이름}: {n}건'(name_count_ed4).
판 3 요청은 같은 입력에서 종전 글 그대로다(얼린 정답 tests/golden_frozen 도 본다).
"""
from __future__ import annotations

import copy
import json

from app.narrative import fallback, generator
from app.narrative.grounding import verify_grounding
from app.narrative.jobs import parse_request
from app.narrative.required_items import Ctx
from app.narrative.wording import class_descriptions
from tests.p4_example_ed4 import MULTI_EXAMPLE_PATHS, as_edition


def _load(task: str) -> dict:
    return json.loads(MULTI_EXAMPLE_PATHS[task].read_text(encoding="utf-8"))


def _prep(raw: dict):
    req, _ = parse_request(copy.deepcopy(raw))
    return req, generator._prepare(req)


def _conclusions(raw: dict) -> dict:
    return {c.item_id: c for c in _prep(raw)[1][3]}


def _ctx(raw: dict) -> Ctx:
    req, (derived, refs, *_rest) = _prep(raw)
    return Ctx(fs=req.fact_sheet, task_type="multiclass", client_inputs=req.client_inputs, derived=derived, benchmark_refs=refs,
               edition=req.report_edition, descriptions=class_descriptions(req.client_inputs))


def _intro(raw: dict, task: str) -> str:
    req, (derived, refs, *_rest) = _prep(raw)
    inp = fallback._input(req.fact_sheet, refs, derived, task_type=task, client_inputs=req.client_inputs, conclusions=None,
                          edition=req.report_edition)
    return fallback._distribution_intro(inp)


# ── 결정 4: 건수 쉼표 ────────────────────────────────────────────────────────────────


def test_d4_edition4_rule_sentences_group_counts_edition3_does_not():
    raw = _load("multilabel")
    c4, c3 = _conclusions(raw), _conclusions(as_edition(raw, 3))
    assert c4["7.1-01"].template_text == (
        "레이블별 혼동행렬을 합친 레이블 칸 12,838칸 중 2,687칸(20.9%)이 틀렸다. "
        "있어야 할 레이블을 놓친 칸(FN)은 1,648칸, 없는 레이블을 붙인 칸(FP)은 1,039칸이다."
    )
    assert c4["7.1-09"].template_text == "예측한 레이블은 모두 3,290개, 실제 레이블은 3,899개로, 레이블을 실제보다 609개 적게 붙였다."
    # 판 3 은 종전 글(쉼표 없음).
    assert "12838칸 중 2687칸" in c3["7.1-01"].template_text
    assert c3["7.1-09"].template_text == "예측한 레이블은 모두 3290개, 실제 레이블은 3899개로, 레이블을 실제보다 609개 적게 붙였다."


def test_d4_fallback_distribution_total_and_counts_are_grouped():
    """폴백 5.3(7.2 칸) 분포 머리 — 표본을 키운 요청(1,000 이상)에서 총수·건수에 쉼표, 판 3 은 그대로."""
    raw = _load("multiclass")
    big = copy.deepcopy(raw)
    dist = big["fact_sheet"]["distribution"]["class_distribution"]
    for k in dist:
        dist[k] *= 30
    big["fact_sheet"]["n_samples"] = sum(dist.values())
    text4, text3 = _intro(big, "multiclass"), _intro(as_edition(big, 3), "multiclass")
    total = sum(dist.values())
    assert total >= 1000
    assert f"총 {total:,}건이며" in text4 and f"총 {total}건이며" in text3
    assert f"2: {dist['2']:,}건(" in text4 and f"2 {dist['2']}건(" in text3


def test_d4_error_ratio_template_groups_counts():
    from app.narrative.required_items import error_ratio_sentence

    req, _ = _prep(_load("multilabel"))
    er = req.fact_sheet.error_ratios
    assert er.misclassified == 2687
    assert error_ratio_sentence(er, edition=4).startswith("틀린 레이블 칸 2,687칸 중")
    assert error_ratio_sentence(er, edition=3).startswith("틀린 레이블 칸 2687칸 중")


def test_d4_grouped_count_passes_grounding_and_a_made_up_one_does_not():
    """허용 목록은 쉼표를 떼고 견준다 — 사실의 2,687 은 받고, 지어낸 2,688 은 위반(헛도는 시험이 아니다: 쉼표 꼴 토큰을 실제로 센다)."""
    raw = _load("multilabel")
    _req, (_derived, _refs, whitelist, *_rest) = _prep(raw)
    ok = verify_grounding(["틀린 레이블 칸은 2,687칸이고 실제 레이블은 3,899개다."], whitelist, edition=4)
    assert ok.passed and ok.checked == 2
    bad = verify_grounding(["틀린 레이블 칸은 2,688칸이다."], whitelist, edition=4)
    assert not bad.passed and bad.violations == ["2,688"]


# ── 결정 5: 클래스 이름 ──────────────────────────────────────────────────────────────


def test_d5_name_with_parenthesized_description_uses_square_brackets():
    raw = _load("multiclass")
    ctx4, ctx3 = _ctx(raw), _ctx(as_edition(raw, 3))
    assert ctx4.name("7") == "7[숫자 7(가로줄 없는 필체 포함)]"
    assert ctx4.name("1") == "1(숫자 1)" and ctx4.name("9") == "9"
    assert ctx4.with_josa(ctx4.name("7"), "은/는") == "7[숫자 7(가로줄 없는 필체 포함)]은"
    assert ctx4.with_josa(ctx4.name("2"), "과/와") == "2와"
    # 판 3 은 종전 꼴·종전 조사.
    assert ctx3.name("7") == "7(숫자 7(가로줄 없는 필체 포함))"
    assert ctx3.with_josa(ctx3.name("7"), "은/는") == "7(숫자 7(가로줄 없는 필체 포함))은"


def test_d5_fallback_distribution_names_digit_end_colon():
    raw = _load("multiclass")
    text4, text3 = _intro(raw, "multiclass"), _intro(as_edition(raw, 3), "multiclass")
    for part in ("0(숫자 0) 54건(10.0%)", "1(숫자 1) 55건(10.2%)", "2: 53건(9.8%)", "7[숫자 7(가로줄 없는 필체 포함)] 54건(10.0%)"):
        assert part in text4, part
    assert "7(숫자 7(가로줄 없는 필체 포함)) 54건" in text3 and "2 53건" in text3
