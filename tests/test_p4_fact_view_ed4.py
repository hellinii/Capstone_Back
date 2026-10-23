"""tests/test_p4_fact_view_ed4.py — 판 4 요청의 사실 시트 보기(판 4 라운드 S2 ④, app.narrative.fact_view_ed4).

판 3 보기는 얼린 정답(tests/golden_frozen/edition3/*.fact_view.json)이 지킨다. 여기서는 판 4 갈래만 본다.
- 판 4 표기: 판정과 이어진 값은 프론트 *_text 그대로, 그 밖은 셋째 자리 끝 0 유지, 비율은 %. 판 3 넷째 자리 표기는 없다.
- 판정 여유: near_target(판 4 뜻)을 읽기만 한다 — 다시 계산하지 않고, 고정 문구 18 유무와 무관하다(계획 9절).
- 통과율 없음, 참조 범위는 수치 위치만(Hamming Loss 처럼 낮을수록 좋은 지표도), 문제의 절은 판 4 자리, 근거 숫자는 판 4 표기.
- 지시문 덜어 내기(TRIM_STEPS)가 판 4 보기에서도 같은 키로 표를 찾는다.

입력은 판 4 임시 예시(tests/contract/fact_sheet_binary_ed4.json — bc_strong, RPT-2026-0001 입력)와 그것을 고친 사본이다.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from app.narrative import fact_view, fact_view_ed4, generator
from app.narrative.schemas import NarrativeRequest
from app.narrative.sections import EDITION4_PLACES, place

CONTRACT = Path(__file__).parent / "contract"
SLOTS = json.loads((CONTRACT / "narrative_slots_ed4.json").read_text(encoding="utf-8"))
MARGIN = fact_view_ed4.MARGIN_KEY


def _raw() -> dict:
    return json.loads((CONTRACT / "fact_sheet_binary_ed4.json").read_text(encoding="utf-8"))


def _view(raw: dict) -> dict:
    req = NarrativeRequest.model_validate(raw)
    ctx = generator._prepare(req)[4]
    assert ctx.edition == req.report_edition
    return ctx.view


def _rows(view: dict) -> dict[str, dict]:
    return {r["지표"]: r for r in view["지표"]}


def _text(view: dict) -> str:
    return json.dumps(view, ensure_ascii=False)


VIEW = _view(_raw())


# ── 판 4 자리 ─────────────────────────────────────────────────────────────────────


def test_edition4_places_match_the_contract_copy():
    want = {s["slot"]: s["edition4_place"] for s in SLOTS["slots"]}
    want |= {s["slot"]: s["edition4_place"] for s in SLOTS["recommend_sections"]}
    assert EDITION4_PLACES == want and len(want) == 9
    assert place("8.2", edition=4) == "5.1" and place("9.1", edition=4) == "6.1"
    assert place("8.2", edition=3) == "8.2" and place("9.1", edition=3) == "9.1"
    assert place("4.3", edition=4) == "4.3"  # 칸 이름이 아니면 그대로


def test_unknown_edition_is_refused():
    req = NarrativeRequest.model_validate(_raw())
    with pytest.raises(ValueError):
        fact_view.build_fact_view(
            req.fact_sheet, {}, [], task_type="binary", task_label="이진 분류", report_purpose="external", edition=5,
        )


def test_edition_decides_the_branch():
    """같은 판 4 요청을 판 3 으로 보면 판 3 보기(넷째 자리·통과율·근소 차이 칸)다 — 판 갈래가 실제로 갈린다."""
    req = NarrativeRequest.model_validate(_raw())
    derived, refs = generator._prepare(req)[:2]
    kw = dict(task_type="binary", task_label="이진 분류", report_purpose="external")
    ed3 = fact_view.build_fact_view(req.fact_sheet, derived, refs, edition=3, **kw)
    ed4 = fact_view.build_fact_view(req.fact_sheet, derived, refs, edition=4, **kw)
    assert ed4 == VIEW
    assert "통과율(참고, %)" in ed3["판정"] and "통과율(참고, %)" not in ed4["판정"]
    assert ed3 != ed4


# ── 지표 표 ──────────────────────────────────────────────────────────────────────


def test_metric_rows_use_edition4_texts():
    rows = _rows(VIEW)
    assert len(rows) == 8
    want = {
        # 지표: (값, 목표값, 결과, 차이, 판정 여유, 구간)
        "Accuracy": ("0.982", "0.950", "충족", "0.032", "작음", "0.9497~0.994"),
        "Recall": ("0.984", "0.970", "충족", "0.014", "작음", "0.917~0.997"),
        "F1 Score": ("0.977", "0.950", "충족", "0.027", "작음", "0.946~1.000"),
        "AUROC": ("0.999", "0.980", "충족", "0.019", "충분", "0.997~1.000"),
        "Log Loss": ("0.053", "0.100", "충족", "0.047", "충분", "0.033~0.079"),
    }
    for name, (value, target, result, margin, small, ci) in want.items():
        r = rows[name]
        got = (r["값"], r["목표값"], r["결과"], r["목표와의 차이(0 이상이면 충족 쪽)"], r[MARGIN], r["95% 신뢰구간"])
        assert got == (value, target, result, margin, small, ci), name
    for name in ("Precision", "AUPRC", "Imbalance Ratio"):  # 참고 지표 — 판정 여유·차이·구간 없음
        r = rows[name]
        assert (r["목표값"], r["결과"]) == ("없음(참고 지표)", "참고 지표"), name
        assert MARGIN not in r and "95% 신뢰구간" not in r, name
    assert (rows["Precision"]["값"], rows["AUPRC"]["값"], rows["Imbalance Ratio"]["값"]) == ("0.969", "0.999", "1.672")


def test_no_edition3_notation_in_edition4_view():
    text = _text(VIEW)
    for word in ("근소 차이", "통과율", "아슬아슬", "정보 제공", "우수", "미흡", "기준 범위 내", "평가(방향 반영)"):
        assert word not in text, word
    # 넷째 자리 이상 숫자는 판정 때문에 늘린 표기뿐이다(이 예시는 Accuracy 하한 0.9497 과 F1 최대 임계값 0.6166).
    long = sorted({t for t in re.findall(r"\d+\.\d{4,}", text)})
    assert long == ["0.6166", "0.9497"]
    # 판 3 표기(예: 0.9767·1.6719·0.0076·0.0325)는 하나도 없다.
    for old in ("0.9767", "1.6719", "0.0076", "0.0325", "0.9825", "0.3333"):
        assert old not in text, old


def test_judgment_margin_is_read_not_recomputed():
    """판정 여유는 near_target 을 그대로 읽는다 — 구간·3% 로 다시 정하지 않는다(판정 지점은 프론트 한 곳)."""
    raw = _raw()
    m = {x["metric_id"]: x for x in raw["fact_sheet"]["metrics"]}
    # AUROC 은 구간이 목표와 떨어졌지만 near_target 을 참으로 보내면 '작음', Accuracy 는 구간이 목표를 포함해도 거짓이면 '충분'.
    m["M9"]["near_target"] = True
    m["M1"]["near_target"] = False
    rows = _rows(_view(raw))
    assert rows["AUROC"][MARGIN] == "작음" and rows["Accuracy"][MARGIN] == "충분"


def test_judgment_margin_without_fixed_sentence_18_and_with_it():
    """판 4 요청은 고정 문구 18 을 싣지 않는다. 18 이 없어도(옛 갈래로 빠지지 않고), 있어도 보기는 같다."""
    raw = _raw()
    assert all(s["id"] != "18" for s in raw["fact_sheet"]["facts"]["fixed_sentences"])
    with18 = copy.deepcopy(raw)
    with18["fact_sheet"]["facts"]["fixed_sentences"].append(
        {"id": "18", "section": "8.3", "text": "목표값과의 차이가 목표값의 3% 미만인 지표(근소 차이): Recall(M3) 0.984(목표 0.970 이상, 충족)"}
    )
    assert _view(with18) == VIEW
    assert [r["지표"] for r in VIEW["지표"] if r.get(MARGIN) == "작음"] == ["Accuracy", "Recall", "F1 Score"]


def test_ci_without_texts_is_printed_by_the_ci_bound_rule():
    """판 4 표기 칸이 버려졌으면(형식 오류) 같은 규칙으로 찍는다 — 넷째 자리 엔진 값 0.9497 → 목표 0.95 와 대소가 같게."""
    raw = _raw()
    for x in raw["fact_sheet"]["metrics"]:
        x["ci_low_text"] = x["ci_high_text"] = None
    rows = _rows(_view(raw))
    assert rows["Accuracy"]["95% 신뢰구간"] == "0.9497~0.994"
    assert rows["F1 Score"]["95% 신뢰구간"] == "0.946~1.000"


# ── 그 밖의 칸 ───────────────────────────────────────────────────────────────────


def test_verdict_has_counts_but_no_pass_rate():
    assert VIEW["판정"] == {
        "종합 판정": "최종 합격", "핵심 지표를 정한 쪽": "의뢰자가 지정함", "판정 대상 지표 수": 5, "목표를 충족한 지표 수(참고)": 5,
    }


def test_confusion_and_distribution_percent():
    totals = VIEW["혼동행렬"]["합계"]
    assert (totals["맞음 비율"], totals["틀림 비율"], totals["FN 비율"]) == ("98.2%", "1.8%", "0.6%")
    assert not any(k.endswith("(%)") for k in totals)
    dist = VIEW["분포·불균형"]
    assert dist["백분율(%, 분모는 평가 표본 수)"] == {"benign": "62.6%", "malignant": "37.4%"}
    assert dist["불균형 비율"] == "1.672"
    effect = dist["불균형 영향(평가 엔진 판단)"]
    assert (effect["정확도 − 매크로 F1"], effect["다수 − 소수 클래스 F1 차이"]) == ("0.001", "0.009")
    assert VIEW["오분류의 성격"]["경계 오류(결정 경계와의 거리가 기준 미만)"] == {"건수": 1, "비율": "33.3%"}


def test_threshold_gain_has_recall_precision_and_risk_side():
    g = VIEW["임계값 조정 여지"]
    assert g["현재 결정 임계값"] == "0.5"  # 의뢰자가 준 글자 그대로(0.500 아님)
    assert g["F1 이 가장 높은 결정 임계값"] == "0.6166"
    assert (g["현재 F1"], g["그 임계값의 F1"], g["향상폭(이 데이터에서 오를 수 있는 최대폭)"]) == ("0.977", "0.984", "0.008")
    assert (g["현재 Recall"], g["그 임계값의 Recall"], g["현재 Precision"], g["그 임계값의 Precision"]) == (
        "0.984", "0.984", "0.969", "0.984",
    )
    assert g["의뢰자가 더 위험하다고 밝힌 오류 쪽 지표"] == {
        "오류": "FN(놓침)", "지표": "Recall", "현재 목표 충족": "예", "F1 이 가장 높은 결정 임계값에서 목표 충족": "예",
    }


def test_facts_use_edition4_texts():
    f = VIEW["서술용 파생값"]
    w = f["F1 이 가장 낮은 클래스"]
    assert (w["F1"], w["비율"], w["매크로 F1"], w["매크로 F1 과의 차이"]) == ("0.977", "37.4%", "0.981", "0.005")
    assert f["의뢰자가 더 위험하다고 밝힌 오류"]["전체 오류 중 비율"] == "33.3%"
    pr = f["양성 클래스 Precision 과 Recall"]
    assert (pr["Precision"], pr["Recall"], pr["차이(Precision − Recall)"]) == ("0.969", "0.984", "-0.015")
    shift = {r["클래스"]: r for r in f["학습↔평가 분포 차이"]}
    assert len(shift) == 2
    assert (shift["benign"]["학습 비율"], shift["benign"]["평가 비율"], shift["benign"]["차이(평가 − 학습)"]) == (
        "62.8%", "62.6%", "−0.2%p",
    )
    assert shift["malignant"]["차이(평가 − 학습)"] == "+0.2%p"


def test_distribution_shift_reads_the_edition4_only_field():
    """평가 − 학습 칸은 판 4 전용 칸을 읽는다(학습 − 평가 difference_text 를 뒤집어 쓰지 않는다)."""
    raw = _raw()
    rows = raw["fact_sheet"]["facts"]["distribution_shift"]
    rows[0]["evaluation_minus_training_text"] = "-0.003"  # 칸을 바꾸면 보기도 따라온다(다른 칸을 읽으면 그대로일 것)
    shift = {r["클래스"]: r for r in _view(raw)["서술용 파생값"]["학습↔평가 분포 차이"]}
    assert shift[rows[0]["label"]]["차이(평가 − 학습)"] == "−0.3%p"


def test_benchmarks_are_neutral_positions():
    b = VIEW["벤치마크(내부 참조 기준)"]
    rows = {r["지표"]: r for r in b["지표별"]}
    assert len(rows) == 6
    assert rows["Accuracy"] == {
        "지표": "Accuracy", "모델 값": "0.982", "참조 범위": "0.85~0.92", "위치": "참조 범위보다 높음",
        "목표값": "0.950", "목표값의 위치": "참조 범위보다 높음",
    }
    assert rows["Precision"]["참조 범위"] == "0.80~0.92" and rows["Precision"]["모델 값"] == "0.969"
    assert all(r["위치"] == "참조 범위보다 높음" for r in rows.values())
    assert all(set(r) <= {"지표", "모델 값", "참조 범위", "위치", "목표값", "목표값의 위치"} for r in rows.values())


def _multilabel_raw(hamming: float) -> dict:
    """계약 다중 레이블 예시(Hamming Loss 목표 ≤ 0.6)를 판 4 요청으로 — Hamming Loss 값만 바꾸고 판정 표기를 맞춘다."""
    from app.narrative.wording_ed4 import ed4_judged

    raw = json.loads((CONTRACT / "fact_sheet_multilabel.json").read_text(encoding="utf-8"))
    raw["report_edition"] = 4
    [m] = [m for m in raw["fact_sheet"]["metrics"] if m["display_name"] == "Hamming Loss"]
    assert m["threshold"] == 0.6 and m["higher_is_better"] is False
    m["value"] = hamming
    m["status"] = "pass"
    m["value_text"] = ed4_judged(hamming, 0.6, False, True)
    m["margin"] = 0.6 - hamming
    m["margin_text"] = ed4_judged(0.6 - hamming, 0, True, True)
    return raw


@pytest.mark.parametrize(
    ("hamming", "position", "shown"),
    [(0.03, "참조 범위보다 낮음", "0.030"), (0.25, "참조 범위보다 높음", "0.250"), (0.1, "참조 범위 안", "0.100")],
)
def test_lower_is_better_metric_position_is_numeric(hamming, position, shown):
    """Hamming Loss(낮을수록 좋음, 참조 범위 0.05~0.20) — 위치는 수치로만 말한다(낮으면 '낮음', 우수라 하지 않는다)."""
    rows = {r["지표"]: r for r in _view(_multilabel_raw(hamming))["벤치마크(내부 참조 기준)"]["지표별"]}
    assert rows["Hamming Loss"]["위치"] == position
    assert rows["Hamming Loss"]["모델 값"] == shown
    assert rows["Hamming Loss"]["참조 범위"] == "0.05~0.20"


def test_problems_use_edition4_places():
    problems = VIEW["문제 목록"]
    assert [(p["번호"], p["우선순위"], p["드러난 절"], p["권고 절"]) for p in problems] == [
        ("P1", "중간", "5.5절", "6.2절"),
        ("P2", "낮음", "5.1절", "6.1절"),
    ]
    assert problems[0]["근거"] == {"Recall": {"값": "0.984", "목표값": "0.970", "목표와의 차이": "0.014"}}
    assert problems[1]["근거"] == {"학습 데이터와의 관계": "같은 출처"}


def _with_problem(problem: dict, **metric_updates) -> dict:
    raw = _raw()
    raw["fact_sheet"]["problems"] = [problem]
    for x in raw["fact_sheet"]["metrics"]:
        if x["metric_id"] in metric_updates:
            x.update(metric_updates[x["metric_id"]])
    return raw


def test_evidence_metric_values_follow_the_judged_text():
    """미달 0.9696(목표 0.97)을 plain 으로 찍으면 0.970 — 충족으로 읽힌다. 근거도 그 지표의 판정 표기를 쓴다."""
    raw = _with_problem(
        {"id": "P1", "kind": "core_fail", "priority": "HIGH", "title": "핵심 지표 미달 (Recall)", "source_section": "8.2",
         "recommend_section": "9.2", "evidence": {"Recall": {"value": 0.9696, "target": 0.97}}, "note": None},
        M3={"value": 0.9696, "status": "fail", "value_text": "0.9696", "margin": -0.0004, "margin_text": "-0.0004",
            "near_target": True},
    )
    raw["fact_sheet"]["verdict"] = "FAIL"
    [p] = _view(raw)["문제 목록"]
    assert p["근거"] == {"Recall": {"값": "0.9696", "목표값": "0.970"}}
    assert (p["드러난 절"], p["권고 절"]) == ("5.1절", "6.2절")


def test_evidence_threshold_tuning_keeps_korean_keys_only():
    gain = _raw()["fact_sheet"]["f1_threshold_gain"]
    raw = _with_problem(
        {"id": "P1", "kind": "threshold_tuning", "priority": "MEDIUM", "title": "임계값 부적합", "source_section": "8.3",
         "recommend_section": "9.2", "evidence": {**gain, "max_f1_raw": 0.984375, "gain_raw": 0.0076308}, "note": None},
    )
    [p] = _view(raw)["문제 목록"]
    assert p["근거"] == {
        "곡선 위 최대 F1": "0.984", "현재 F1": "0.977", "향상폭": "0.008", "최대 F1 의 결정 임계값": "0.6166", "임계값 부적합": "아니오",
    }


def test_evidence_imbalance_and_weakest_and_shift():
    raw = _raw()
    fs = raw["fact_sheet"]
    ie = fs["imbalance_effect"]
    fs["problems"] = [
        {"id": "P1", "kind": "imbalance_effect", "priority": "MEDIUM", "title": "불균형 영향 있음", "source_section": "7.2",
         "recommend_section": "9.1", "evidence": {**ie, "imbalance_ratio_raw": 1.671875}, "note": None},
        {"id": "P2", "kind": "weakest_class", "priority": "LOW", "title": "취약 클래스 (malignant)", "source_section": "7.1",
         "recommend_section": "9.1", "evidence": {"malignant": {
             "f1": 0.9767441860465116, "support": 64, "share": 0.374269006, "macro_f1": 0.981329839502129,
             "gap_from_macro": 0.004585654, "low_support": False}}, "note": None},
        {"id": "P3", "kind": "distribution_shift", "priority": "LOW", "title": "분포 차이 (benign)", "source_section": "7.2",
         "recommend_section": "9.1", "evidence": {"benign": {
             "training_ratio": 0.628140704, "evaluation_ratio": 0.625730994, "difference": 0.002409709}}, "note": None},
    ]
    p1, p2, p3 = _view(raw)["문제 목록"]
    assert p1["근거"]["불균형 비율"] == "1.672" and p1["근거"]["다수 − 소수 F1 차이"] == "0.009"
    assert not any(k.endswith("_raw") or k == "display" for k in p1["근거"])
    assert p2["근거"] == {"malignant": {
        "F1": "0.977", "표본 수": 64, "비율": "37.4%", "매크로 F1": "0.981", "매크로 F1 과의 차이": "0.005", "표본이 작음": "아니오"}}
    assert p3["근거"] == {"benign": {"학습 비율": "62.8%", "평가 비율": "62.6%", "차이": "0.002"}}
    assert [(p["드러난 절"], p["권고 절"]) for p in (p1, p2, p3)] == [("5.3절", "6.1절"), ("5.2절", "6.1절"), ("5.3절", "6.1절")]


# ── 덜어 내기 ────────────────────────────────────────────────────────────────────


def test_trim_steps_find_tables_in_edition4_view():
    """지시문이 상한을 넘으면 같은 차례로 표를 덜어 낸다 — 판 4 보기도 같은 키라 단계마다 실제로 덜어진다."""
    for name, step in fact_view.TRIM_STEPS:
        trial = copy.deepcopy(VIEW)
        assert step(trial), name
        assert trial != VIEW and trial["덜어 낸 표(일부만 실림)"], name
