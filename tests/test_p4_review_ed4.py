"""tests/test_p4_review_ed4.py — 판 4 라운드 S2·S3 검토(2026-10-24) 반영의 백엔드 쪽.

결정 D1~D9(사용자 답변·오케스트레이터 결정 — 두 저장소 같은 규칙, 공통 대조 표 number_wording_ed4.json 에 사례 줄)와 검토
지적(T4·T6·T7·T9·T10·T11 등)이 판 4 사실 보기·규칙 문장·지시문·숫자 허용 목록에 실제로 닿는지 본다. 표의 줄 자체는
test_p4_wording_ed4.py(백엔드 함수)·test_p4_contract_ed4.py(기준 구현)가 본다.

- D1 목표값: 넷째 자리 이상 목표는 받은 글자 그대로(0.9501 — 셋째 자리 0.950 으로 줄이지 않는다).
- D2 같은 수는 같은 글자: 최약 클래스 F1 은 프론트가 실은 facts.weakest_class.f1_text 를 읽고, 클래스별 표·평균 줄 등은 판정 대상
  지표와 같은 원값이면 그 value_text.
- D3 최약 클래스가 셋째 자리로 다른 클래스와 같아 보이면 늘려 찍고, 동점은 여섯째 자리로도 같을 때만(판 3 은 넷째 자리 그대로).
- D4 지연시간은 4.8 표와 같은 둘째 자리(toFixed(2)).
- D6 문제 근거 숫자는 판 4 글자로 읽고, 임계값 조정 근거에 섞인 판 4 새 칸은 버린다.
- D7 best_threshold_text 는 숫자 꼴(지수 꼴이면 그 칸만 버리고 판 4 given 으로 — 고정소수).
판 3 출력이 그대로인 것은 얼린 정답(tests/golden_frozen)이 본다.
"""
from __future__ import annotations

import copy
import json

import pytest

from app.narrative import checks, generator
from app.narrative.fact_view_ed4 import MARGIN_KEY
from app.narrative.grounding import verify_grounding
from app.narrative.jobs import DISPLAY_DROPPED, parse_request
from app.narrative.required_items import weakest_ties
from tests.conftest import requested_fields
from tests.p4_example_ed4 import EXAMPLE_NEAR_INTERPRET, MULTI_EXAMPLE_PATHS, as_edition, example_raw


def _load(task: str) -> dict:
    return json.loads(MULTI_EXAMPLE_PATHS[task].read_text(encoding="utf-8"))


def _prep(raw: dict):
    req, repaired = parse_request(copy.deepcopy(raw))
    return req, generator._prepare(req), repaired


def _view(raw: dict) -> dict:
    return _prep(raw)[1][4].view


def _conclusions(raw: dict) -> dict:
    return {c.item_id: c for c in _prep(raw)[1][3]}


def _wl(raw: dict) -> set:
    return _prep(raw)[1][2]


def _metric(raw: dict, mid: str) -> dict:
    return next(m for m in raw["fact_sheet"]["metrics"] if m["metric_id"] == mid)


def _ci_sentence(raw: dict) -> str:
    """5.1(8.2 칸) 신뢰구간 해석 결론 — '95% 신뢰구간이 목표값을 포함하는 지표는 …'."""
    return next(c.text for c in _conclusions(raw).values() if "95% 신뢰구간이 목표값을 포함하는 지표는" in c.text)


def _rows(view: dict) -> dict:
    return {r["지표"]: r for r in view["지표"]}


# ── D1 목표값 표기 ────────────────────────────────────────────────────────────────────


def _target_4th(raw: dict, mid: str, target: float) -> dict:
    out = copy.deepcopy(raw)
    _metric(out, mid)["threshold"] = target
    return out


def test_d1_fourth_digit_target_is_printed_as_received_everywhere():
    """넷째 자리 목표 0.9501(Accuracy — 구간 0.9497~0.994 가 여전히 포함, 값 0.982 충족)은 4.3 표·5.1 신뢰구간 문장·5.4 목표값
    수준·사실 보기·문제 근거 모두 0.9501 로 — 셋째 자리 0.950 은 넷째 자리 목표와 판정이 어긋나 보일 수 있다."""
    raw = _target_4th(example_raw(), "M1", 0.9501)
    view = _view(raw)
    assert _rows(view)["Accuracy"]["목표값"] == "0.9501"
    assert _rows(view)["Recall"]["목표값"] == "0.970"  # 셋째 자리 이하 목표는 그대로 셋째 자리
    cons = _conclusions(raw)
    assert "Accuracy(구간 0.9497~0.994, 목표 0.9501)" in _ci_sentence(raw)
    assert "Accuracy(목표 0.9501)" in cons["8.1-02"].text
    bench = {r["지표"]: r for r in view["벤치마크(내부 참조 기준)"]["지표별"]}
    assert bench["Accuracy"]["목표값"] == "0.9501"
    assert "0.9501" in _wl(raw)


def test_d1_fourth_digit_target_in_b20_and_failed_metric_lists():
    raw = _target_4th(example_raw(), "M3", 0.9701)  # Recall — 위험 쪽 지표(B20)
    g = raw["fact_sheet"]["f1_threshold_gain"]
    g.update(gain=0.12, notable=True, gain_text="0.120", current_f1_text="0.864")
    assert "이 값은 목표 0.9701을 충족한다." in _conclusions(raw)["8.2-06"].text
    # 문제 근거의 목표(미달 지표 목록은 아래 규칙 문장 시험).
    ml = _target_4th(_load("multilabel"), "M4", 0.4512)
    view = _view(ml)
    problem = next(p for p in view["문제 목록"] if p["번호"] == "P1")
    assert problem["근거"]["F1 Score"]["목표값"] == "0.4512"


async def test_d1_fourth_digit_target_in_rule_sentences():
    ml = _target_4th(_load("multilabel"), "M4", 0.4512)
    req, _ = parse_request(copy.deepcopy(ml))
    parts = (await generator.generate_sections(None, req)).response.model_dump(mode="json")
    assert "F1 Score(0.399, 목표 0.4512)" in parts["conclusion"]["narrative"]
    assert "F1 Score(0.399, 목표 0.4512)" in parts["conclusion"]["risks"]


# ── D2·D3 최약 클래스 F1 과 같은 수의 글자 ─────────────────────────────────────────────


def test_d2_weakest_f1_text_from_the_frontend_is_read_everywhere():
    """프론트가 실은 최약 클래스 F1 글자를 사실 보기(최약 클래스·클래스별 표)·규칙 문장(5.2·5.5)이 그대로 쓴다 — 백엔드가 다시
    정하지 않는다(여기서는 다른 규칙이 낼 수 없는 글자 0.97674 로 본다)."""
    raw = example_raw()
    raw["fact_sheet"]["facts"]["weakest_class"]["f1_text"] = "0.97674"
    view = _view(raw)
    assert view["서술용 파생값"]["F1 이 가장 낮은 클래스"]["F1"] == "0.97674"
    per_class = {r["클래스"]: r for r in view["클래스별 성능"]["클래스별"]}
    assert per_class["malignant"]["F1"] == "0.97674"
    cons = _conclusions(raw)
    assert "F1 0.97674인 malignant(악성 종양)" in cons["7.1-10"].text
    assert "F1 0.97674인 malignant(악성 종양)" in cons["8.3-01"].text
    assert "0.97674" in _wl(raw)


def test_d2_same_value_as_a_judged_metric_takes_its_value_text():
    """판정과 무관한 자리의 값이 판정 대상 지표의 원값과 같으면 그 value_text — 이진 양성 클래스 F1 = F1 Score(M4)·Recall = M3.
    f1_text 가 없으면(그 칸을 싣기 전 요청) 같은 규칙으로 찍는다."""
    raw = example_raw()
    _metric(raw, "M4")["value_text"] = "0.9767"  # 판정과 어긋나 보여 늘린 표기라고 친다
    _metric(raw, "M3")["value_text"] = "0.9844"
    del raw["fact_sheet"]["facts"]["weakest_class"]["f1_text"]
    view = _view(raw)
    per_class = {r["클래스"]: r for r in view["클래스별 성능"]["클래스별"]}
    assert (per_class["malignant"]["F1"], per_class["malignant"]["Recall"]) == ("0.9767", "0.9844")
    assert (per_class["benign"]["F1"], per_class["benign"]["Recall"]) == ("0.986", "0.981")  # 같은 원값이 아니면 plain
    assert view["서술용 파생값"]["F1 이 가장 낮은 클래스"]["F1"] == "0.9767"
    assert "F1 0.9767인 malignant(악성 종양)" in _conclusions(raw)["7.1-10"].text
    # 멀티레이블 매크로 평균 줄 F1·최약 레이블의 매크로 F1 = F1 Score(M4, 매크로).
    ml = _load("multilabel")
    _metric(ml, "M4")["value_text"] = "0.3993"
    view = _view(ml)
    macro = next(a for a in view["클래스별 성능"]["평균 줄"] if a["평균"] == "매크로 평균")
    assert macro["F1"] == "0.3993"
    assert view["서술용 파생값"]["F1 이 가장 낮은 레이블"]["매크로 F1"] == "0.3993"


def test_d3_multiclass_weakest_looks_different_from_the_runner_up():
    """다중 예시(RPT-2026-0003): 9(0.97196)와 5(0.97248)는 셋째 자리로 둘 다 0.972 — 넷째 자리로 늘려 9 하나가 최약이다
    (S2·S3 검토 T2 를 D3 로 정함). 클래스별 표의 5 도 0.9725 로 늘려 같은 글자로 보이지 않게 한다."""
    raw = _load("multiclass")
    assert raw["fact_sheet"]["facts"]["weakest_class"]["f1_text"] == "0.9720"
    req = parse_request(copy.deepcopy(raw))[0]
    assert weakest_ties(req.fact_sheet, edition=4) == ([], False)
    cons = _conclusions(raw)
    assert cons["7.1-10"].text.startswith("F1 이 가장 낮은 클래스는 F1 0.9720인 9이며,")
    assert cons["8.3-01"].text.startswith("가장 약한 클래스는 F1 0.9720인 9이다.")
    per_class = {r["클래스"]: r for r in _view(raw)["클래스별 성능"]["클래스별"]}
    assert (per_class["9"]["F1"], per_class["5"]["F1"], per_class["6"]["F1"]) == ("0.9720", "0.9725", "0.981")


def _near_tie(raw: dict, gap: float) -> dict:
    """다중 예시에서 5 의 F1 을 9 보다 gap 만큼 크게 둔 사본(f1_text 는 지워 같은 규칙으로 찍게)."""
    out = copy.deepcopy(raw)
    w = out["fact_sheet"]["facts"]["weakest_class"]
    del w["f1_text"]
    for p in out["fact_sheet"]["per_class"]:
        if p["label"] == "5":
            p["f1"] = w["f1"] + gap
    return out


def test_d3_ties_are_only_ties_at_the_sixth_digit_in_edition4():
    """판 4 동점은 여섯째 자리로도 같을 때만 — 판 3 은 넷째 자리 동점 그대로. 같은 입력이 판마다 갈린다(판을 넘기는 자리가 빠지면
    판 4 글에 '같은 두 클래스' 가 샌다)."""
    close = _near_tie(_load("multiclass"), 3e-5)  # 0.97196 · 0.97199 — 넷째 자리 0.9720 로 같고 다섯째 자리로 다르다
    req = parse_request(copy.deepcopy(close))[0]
    assert weakest_ties(req.fact_sheet, edition=3) == (["9", "5"], False)
    assert weakest_ties(req.fact_sheet, edition=4) == ([], False)
    cons4 = _conclusions(close)
    assert cons4["7.1-10"].text.startswith("F1 이 가장 낮은 클래스는 F1 0.97196인 9이며,")
    assert cons4["8.3-01"].text.startswith("가장 약한 클래스는 F1 0.97196인 9이다.")
    assert _view(close)["서술용 파생값"]["F1 이 가장 낮은 클래스"].get("F1 이 같은 클래스") is None
    notes = " ".join(_prep(close)[1][4].briefs["7.1"].notes)
    assert "F1 이 같은 최하위" not in notes
    cons3 = _conclusions(as_edition(close, 3) | {"report_edition": 3})
    assert "같은 두 클래스(9, 5)" in cons3["7.1-10"].text
    # 여섯째 자리로도 같으면 판 4 도 동점 — 함께 밝힌다.
    tie = _near_tie(_load("multiclass"), 1e-9)
    assert weakest_ties(parse_request(copy.deepcopy(tie))[0].fact_sheet, edition=4) == (["9", "5"], False)
    assert "F1 0.972로 같은 두 클래스(9, 5)" in _conclusions(tie)["7.1-10"].text


# ── D4 지연시간 ───────────────────────────────────────────────────────────────────────


def _latency(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    out["fact_sheet"]["latency"] = {"available": True, "mean": 12.345, "p50": 0.125, "p95": 125.456, "p99": 1.005, "unit": "ms"}
    return out


def test_d4_latency_is_printed_like_the_4_8_table():
    raw = _latency(example_raw())
    assert _view(raw)["지연시간"] == {"평균": "12.35", "p50": "0.13", "p95": "125.46", "p99": "1.00", "단위": "ms"}
    assert _conclusions(raw)["8.3-04"].text == (
        "지연시간은 평균 12.35ms, 중앙값(p50) 0.13ms, 95번째 백분위수(p95) 125.46ms, 99번째 백분위수(p99) 1.00ms이다."
    )
    wl = _wl(raw)
    assert verify_grounding(["평균 12.35ms, p50 0.13ms, p99 1.00ms"], wl, edition=4).passed
    # 판 3 식 표기(넷째 자리·파이썬 반올림 0.12)는 판 4 글에서 위반이다.
    for text in ("평균 12.345ms", "p50 0.12ms", "p95 125.456ms"):
        assert not verify_grounding([text], wl, edition=4).passed, text
    # 판 3 은 그대로(넷째 자리).
    assert "평균 12.345ms" in _conclusions(as_edition(raw, 3))["8.3-04"].text


def test_d4_declared_latency_number_is_found_in_edition4_notation():
    """D24(대화형 용도의 지연시간 확인 숫자) — 판 4 글자(0.125 → 0.13, 절반에서 0에서 먼 쪽)도 그 숫자로 본다."""
    assert checks.d24_present("p50 은 0.13ms 이다.", (0.125,), edition=4)
    assert not checks.d24_present("p50 은 0.13ms 이다.", (0.125,), edition=3)


# ── D6 문제 근거 ──────────────────────────────────────────────────────────────────────


def test_d6_evidence_in_edition4_text_and_threshold_tuning_picks_engine_keys():
    """프론트가 근거 숫자를 판 4 글자로 보내도(판 3 숫자로 보내도) 사실 보기는 같은 사실 시트의 판 4 글자로 읽고, 임계값 조정 근거는
    엔진 칸 다섯만 골라 판 4 글자로 — 판 4 파서 결과를 펼쳐 섞인 새 칸(*_raw·best_recall·*_text)은 싣지 않는다."""
    raw = example_raw()
    problems = raw["fact_sheet"]["problems"]
    problems[0]["evidence"] = {"Recall": {"value": "0.984", "target": "0.970", "margin": "0.014"}}
    problems.insert(1, {
        "id": "P9", "kind": "threshold_tuning", "priority": "MEDIUM", "title": "결정 임계값 조정 여지", "source_section": "8.2",
        "recommend_section": "9.2", "note": None,
        "evidence": {
            "max_f1": 0.9844, "current_f1": 0.9767, "gain": 0.0076, "best_threshold": 0.6166, "notable": False, "gain_text": "0.0076",
            "best_recall": 0.9844, "best_recall_raw": 0.984375, "max_f1_raw": 0.984375, "best_threshold_text": "0.6166",
        },
    })
    view = _view(raw)
    by_id = {p["번호"]: p for p in view["문제 목록"]}
    assert by_id["P1"]["근거"] == {"Recall": {"값": "0.984", "목표값": "0.970", "목표와의 차이": "0.014"}}
    assert by_id["P9"]["근거"] == {
        "곡선 위 최대 F1": "0.984", "현재 F1": "0.977", "향상폭": "0.008", "최대 F1 의 결정 임계값": "0.6166", "임계값 부적합": "아니오",
    }
    # 판 3 숫자로 보내도 같은 글자.
    problems[0]["evidence"] = {"Recall": {"value": 0.984375, "target": 0.97, "margin": 0.0144}}
    assert {p["번호"]: p for p in _view(raw)["문제 목록"]}["P1"]["근거"] == by_id["P1"]["근거"]


# ── D7 best_threshold_text 는 숫자 꼴 ────────────────────────────────────────────────


def test_d7_exponent_threshold_text_is_dropped_and_printed_fixed_point():
    raw = example_raw()
    g = raw["fact_sheet"]["f1_threshold_gain"]
    g.update(best_threshold=1e-05, best_threshold_text="1e-05")
    req, prepared, repaired = _prep(raw)
    assert DISPLAY_DROPPED in repaired and req.fact_sheet.f1_threshold_gain.best_threshold_text is None
    assert prepared[4].view["임계값 조정 여지"]["F1 이 가장 높은 결정 임계값"] == "0.00001"
    g["best_threshold_text"] = "0.00001"
    req, _, repaired = _prep(raw)
    assert repaired == [] and req.fact_sheet.f1_threshold_gain.best_threshold_text == "0.00001"


def test_d7_given_threshold_has_no_trailing_zero_or_exponent():
    """의뢰자 결정 임계값이 정수 꼴이면 소수점 없이 — 프론트 4.4 머리·고정 문구 9 의 String(1) 과 같은 글자(S2·S3 검토 T15)."""
    raw = example_raw()
    raw["fact_sheet"]["derived_prediction"]["threshold"] = 1
    assert _view(raw)["임계값 조정 여지"]["현재 결정 임계값"] == "1"


# ── T4 5.5 해석 재료(판정 여유) ────────────────────────────────────────────────────────


def test_t4_near_target_interpretation_material_is_exact():
    ctx = _prep(example_raw())[1][4]
    assert [t for t in ctx.briefs["8.3"].interpret if "판정 여유가 작은 지표 가운데" in t][0].endswith(EXAMPLE_NEAR_INTERPRET)


def test_t4_near_target_without_interval_branch():
    raw = example_raw()
    _metric(raw, "M1").update(ci_low=None, ci_high=None, ci_low_text=None, ci_high_text=None, ci_contains_target=None)
    text = [t for t in _prep(raw)[1][4].briefs["8.3"].interpret if "판정 여유가 작은 지표 가운데" in t][0]
    assert text.endswith(
        "해석 재료: 판정 여유가 작은 지표 가운데 목표를 충족한 지표는 Accuracy, Recall, F1 Score이다. 95% 신뢰구간이 목표값을 "
        "포함하는 지표는 Recall, F1 Score이고, 신뢰구간이 없어 목표값과의 차이가 목표값의 3% 미만인 지표는 Accuracy이다. "
        "신뢰구간이 없는 지표는 표본이 달라지면 판정이 뒤집힐 수 있다고 쓰지 말고 판정의 여유가 작다는 사실만 쓴다. 판정 여유가 "
        "작은 지표는 이 지표들뿐이다. 다른 지표를 여유가 작다거나 판정이 뒤집힐 수 있다고 해석하지 않는다."
    )


@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_t4_edition4_user_messages_have_no_edition3_near_words(task):
    """판 4 지시문에는 판 3 의 '근소 차이' 글이 섞이지 않는다(판 4 갈래가 막히면 '근소 차이 지표 가운데…' 가 실린다)."""
    from app.narrative import prompt

    raw = example_raw() if task == "binary" else _load(task)
    req, prepared, _ = _prep(raw)
    ctx = prepared[4]
    for ids in (("7.1", "7.2", "7.3"), ("8.1", "8.2", "8.3"), ("9",)):
        msg = prompt.build_user_message(ctx, ids, regenerate=False)
        assert "근소 차이" not in msg, (task, ids)  # 문제 이름 '핵심 지표 근소 합격' 은 판 4 이름이다
        assert "아슬아슬" not in msg, (task, ids)


# ── T6 판 4 허용 목록에 판 3 변형이 들어가지 않는다 ──────────────────────────────────────


def test_t6_counts_added_by_the_generator_are_exact_in_edition4():
    """생성기가 더하는 건수('둘 다 아닌 오류' 등)는 판 4 에서 찍힌 그대로만 — 1건이 '100%' 를 허용하지 않는다."""
    raw = example_raw()
    er = raw["fact_sheet"]["error_ratios"]
    er["confident"]["count"] = 1
    wl4 = _wl(raw)
    assert not verify_grounding(["오분류의 100%가 경계 오류다."], wl4, edition=4).passed
    # 판 3 은 종전대로 변형까지(같은 사본의 판 3 허용 목록).
    assert verify_grounding(["오분류의 100%가 경계 오류다."], _wl(as_edition(raw, 3)), edition=3).passed


# ── T7 멀티레이블 D49 ② 까닭의 분모 ─────────────────────────────────────────────────────


def test_t7_multilabel_small_sample_reason_names_label_occurrences():
    text = _conclusions(_load("multilabel"))["8.3-01"].text
    assert "작은 표본(30건 미만 또는 레이블 등장의 5% 미만)이고 매크로 F1 보다 0.1 이상 낮아 운영 위험으로 다룬다." in text
    assert "전체의 5%" not in text and "전체 표본의 5%" not in text


# ── T9 미달 지표의 판정 여유 ──────────────────────────────────────────────────────────


def test_t9_failed_metric_with_small_margin_is_small():
    raw = example_raw()
    _metric(raw, "M9").update(status="fail", near_target=True, ci_contains_target=True, margin=-0.001, margin_text="-0.001")
    rows = _rows(_view(raw))
    assert (rows["AUROC"]["결과"], rows["AUROC"][MARGIN_KEY]) == ("미달", "작음")


# ── T10 신뢰구간 경계의 ci_bound 대비 ──────────────────────────────────────────────────


def test_t10_interval_bounds_fall_back_to_ci_bound_not_plain():
    raw = example_raw()
    for m in raw["fact_sheet"]["metrics"]:
        m.update(ci_low_text=None, ci_high_text=None)
    _metric(raw, "M1")["ci_low"] = 0.9496962348678732  # 반올림 전 경계(사실 시트 ci_low 는 숫자 그대로 받는다)
    assert "Accuracy(구간 0.9497~0.994, 목표 0.950)" in _ci_sentence(raw)


# ── T11 rescue 경로의 권고 채우기도 판 4 ───────────────────────────────────────────────


async def test_t11_rescued_recommendation_is_the_edition4_template(make_section_client):
    """9칸 첫 글이 권고 하나를 빠뜨리고 다시 만들기가 실패하면 빠진 권고를 템플릿으로 채운다 — 판 4 면 B24 글이다."""
    import anthropic
    import httpx2

    req = parse_request(example_raw())[0]
    only_p2 = [{"problem_id": "P2", "category": "재평가", "action": "다른 출처 데이터로 다시 평가한다.", "expected_impact": "일반화를 확인한다."}]
    calls = {"9": 0}

    def reply(kwargs):
        if requested_fields(kwargs)[0] == "section_9_1":
            calls["9"] += 1
            if calls["9"] > 1:
                return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
        return None

    fake = make_section_client(
        {"section_9_1": "데이터 측면의 권고 서술이다.", "section_9_2": "모델 측면의 권고 서술이다.", "recommendations": only_p2},
        replies=[reply] * 8,
    )
    gen = await generator.generate_sections(fake, req)
    assert gen.trace["sections"]["9"]["outcome"] == "filled"
    assert gen.trace["recommendations_from_template"] == ["P1"]
    p1 = next(r for r in gen.response.recommendations if r.problem_id == "P1")
    assert p1.action.startswith("판정 여유가 작은 채 합격한 핵심 지표를")  # B24 — 판 3 글은 '목표값에 근접해 합격한'
