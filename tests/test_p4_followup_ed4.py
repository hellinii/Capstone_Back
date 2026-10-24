"""tests/test_p4_followup_ed4.py — 판 4 라운드 S2·S3 후속(2026-10-24)의 백엔드 쪽.

- U1(사용자 답 T3) 멀티레이블 4.5 학습·평가 비율의 분모 = 표본 수. 비율은 프론트가 사실 시트 distribution_shift 에 싣고 백엔드는
  읽기만 한다 — 판 4 사실 보기·7.2-03 규칙 문장의 평가 비율이 4.5 분포 문장(분모 n_samples)과 같은 퍼센트인지, 학습 표본 수가
  없어 비교를 생략한 상태(training_samples_not_provided)가 'B16 학습 분포 없음' 과 다른 이유로 읽히는지 본다.
- U2(오케스트레이터 결정) D3 클래스별 표 — 최약 클래스와 셋째 자리로 같아 보이는 다른 클래스 F1 = distinct(그 값, [최약 F1]).
  공통 대조 표 distinct 의 class_table 줄을 백엔드 클래스별 표 함수(사실 보기·규칙 문장 같은 한 곳)로 본다.
- U3 D2 보강 — 다중 선언 평균(micro F1 = M1 원값)과 평균 차이 other_f1 이 판정 대상 지표와 같은 원값이면 그 value_text.
예시 파일은 그대로 두고 시험 안 사본만 고친다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.narrative import checks, fallback, generator
from app.narrative import required_items as ri
from app.narrative import wording_ed4
from app.narrative.fact_view_ed4 import _class_f1
from app.narrative.grounding import verify_grounding
from app.narrative.jobs import parse_request
from tests.p4_example_ed4 import MULTI_EXAMPLE_PATHS, as_edition, example_raw

TABLE = json.loads((Path(__file__).parent / "contract" / "number_wording_ed4.json").read_text(encoding="utf-8"))


def _load(task: str) -> dict:
    return json.loads(MULTI_EXAMPLE_PATHS[task].read_text(encoding="utf-8"))


def _prep(raw: dict):
    req, _ = parse_request(copy.deepcopy(raw))
    return req, generator._prepare(req)


def _view(raw: dict) -> dict:
    return _prep(raw)[1][4].view


def _conclusions(raw: dict) -> dict:
    return {c.item_id: c for c in _prep(raw)[1][3]}


def _metric(raw: dict, mid: str) -> dict:
    return next(m for m in raw["fact_sheet"]["metrics"] if m["metric_id"] == mid)


def _ctx(req, task: str) -> ri.Ctx:
    return ri.Ctx(fs=req.fact_sheet, task_type=task, client_inputs={}, derived={}, benchmark_refs=[], edition=4,
                  selected={"M22"})


# ── U1 멀티레이블 학습·평가 비율의 분모 = 표본 수 ─────────────────────────────────────────

# yeast_strong(RPT-2026-0005) 평가 표본 917. 학습 쪽은 의뢰자가 학습 표본 수 1500 을 줬다고 친다(예시에는 학습 분포가 없다).
_N_EVAL, _N_TRAIN = 917, 1500
_SHIFT = {  # 레이블: (평가 건수 — 예시 class_distribution 그대로, 학습 건수)
    "Class12": (688, 1000),  # 평가 688/917 = 0.75027 → 75.0%, 학습 1000/1500 → 66.7%, 차 +8.4%p(10%p 미만)
    "Class13": (683, 900),   # 평가 683/917 = 0.74482 → 74.5%, 학습 900/1500 → 60.0%, 차 +14.5%p(10%p 이상)
}


def _multilabel_compared() -> dict:
    """멀티 예시에 표본 수 분모로 낸 학습·평가 비율(프론트가 판 4 에서 싣는 값)을 넣은 사본."""
    raw = _load("multilabel")
    facts = raw["fact_sheet"]["facts"]
    assert raw["fact_sheet"]["n_samples"] == _N_EVAL
    rows = []
    for label, (n_eval, n_train) in _SHIFT.items():
        assert raw["fact_sheet"]["distribution"]["class_distribution"][label] == n_eval
        ev, tr = n_eval / _N_EVAL, n_train / _N_TRAIN
        flagged = abs(tr - ev) >= 0.1
        rows.append({
            "label": label, "training_ratio": tr, "evaluation_ratio": ev, "difference": tr - ev,
            "difference_text": wording_ed4.ed4_signed_judged(tr - ev, 0.1, flagged), "flagged": flagged,
            "evaluation_minus_training_text": wording_ed4.ed4_signed_judged(ev - tr, 0.1, flagged),
        })
    facts["distribution_shift_status"] = "compared"
    facts["distribution_shift"] = rows
    return raw


def test_u1_multilabel_evaluation_ratio_is_the_same_percent_as_the_distribution_sentence():
    """멀티 평가 비율 = 그 레이블이 붙은 평가 표본 / 전체 평가 표본 — 사실 보기 '분포·불균형' 백분율·7.2 분포 머리 문장(분모
    n_samples)과 같은 퍼센트다. 합은 100% 를 넘는다(레이블이 여럿인 표본)."""
    raw = _multilabel_compared()
    view = _view(raw)
    shares = view["분포·불균형"]["백분율(%, 분모는 평가 표본 수)"]
    rows = {r["레이블"]: r for r in view["서술용 파생값"]["학습↔평가 분포 차이"]}
    assert view["서술용 파생값"]["학습↔평가 분포 비교"] == "비교함"
    assert (rows["Class12"]["평가 비율"], rows["Class12"]["학습 비율"], rows["Class12"]["차이(평가 − 학습)"]) == (
        "75.0%", "66.7%", "+8.4%p")
    assert (rows["Class13"]["평가 비율"], rows["Class13"]["학습 비율"], rows["Class13"]["차이(평가 − 학습)"]) == (
        "74.5%", "60.0%", "+14.5%p")
    assert (shares["Class12"], shares["Class13"]) == ("75.0%", "74.5%")  # 같은 숫자
    assert (rows["Class12"]["10%p 이상"], rows["Class13"]["10%p 이상"]) == ("아니오", "예")
    # 분모 안내 — 숫자를 넣지 않는다(허용 목록이 보기 값의 숫자를 받는다).
    note = view["서술용 파생값"]["학습↔평가 비율의 분모"]
    assert note.startswith("표본 수 — ") and not any(ch.isdigit() for ch in note)
    assert sum(float(v.rstrip("%")) for v in shares.values()) > 100
    # 7.2-03 규칙 문장과 7.2 분포 머리 문장(폴백)이 같은 퍼센트.
    cons = _conclusions(raw)
    assert cons["7.2-03"].text == "학습 분포와 평가 분포의 비율 차가 10%p 이상인 레이블은 Class13 학습 60.0%·평가 74.5%이다."
    req, (derived, refs, *_rest) = _prep(raw)
    intro = fallback._distribution_intro(fallback._input(req.fact_sheet, refs, derived, task_type="multilabel",
                                                         client_inputs=req.client_inputs, conclusions=None, edition=4))
    assert "Class12(세포 운명) 688건(75.0%)" in intro and "Class13 683건(74.5%)" in intro
    # 허용 목록은 그 퍼센트를 받고(보기 값), 학습 표본 수 분모가 아닌 비율(레이블 등장 합 분모 — Class12 688/3899 = 17.6%)은 받지 않는다.
    wl = _prep(raw)[1][2]
    assert verify_grounding([cons["7.2-03"].text, intro], wl, edition=4).violations == []
    assert verify_grounding(["Class12 평가 비율은 17.6%이다."], wl, edition=4).violations


def test_u1_denominator_note_is_multilabel_only_and_absent_without_rows():
    """분모 안내는 멀티레이블 비교 행이 있을 때만 — 이진·다중(원래 표본 수 분모)과 비교하지 않은 멀티에는 없다."""
    assert "학습↔평가 비율의 분모" not in _view(_load("multilabel"))["서술용 파생값"]
    mc = _view(_load("multiclass"))["서술용 파생값"]
    assert mc["학습↔평가 분포 비교"] == "비교함" and "학습↔평가 비율의 분모" not in mc


def test_u1_training_samples_not_provided_is_a_different_reason_from_no_training_distribution():
    """학습 분포는 있으나 학습 표본 수가 없으면(멀티 판 4) 비교를 생략한 이유가 'B16 학습 분포 없음' 과 다르다 — 사실 보기가
    다른 이유로 읽고, 7.2-03 결론은 없다(비교하지 않음). 4.5 문장은 B33(초안, 계약 decided)이고 S5 렌더러 몫이다."""
    raw = _load("multilabel")
    raw["fact_sheet"]["facts"]["distribution_shift_status"] = "training_samples_not_provided"
    view = _view(raw)["서술용 파생값"]
    assert view["학습↔평가 분포 비교"] == "학습 표본 수를 받지 않아 비교하지 않음(멀티레이블 비율의 분모는 표본 수)"
    assert "학습 분포를 받지 않아" not in view["학습↔평가 분포 비교"]
    assert "7.2-03" not in _conclusions(raw)
    # 학습 분포가 없는 예시 그대로(not_provided)는 B16 이유 그대로다.
    assert _view(_load("multilabel"))["서술용 파생값"]["학습↔평가 분포 비교"] == "학습 분포를 받지 않아 비교하지 않음"


# ── U2 D3 클래스별 표 — 공통 대조 표 distinct 의 class_table 줄 ─────────────────────────────

_CLASS_ROWS = [c for c in TABLE["number_cases"] if c["rule"] == "distinct" and c.get("class_table")]


def test_u2_table_has_class_table_rows():
    assert len(_CLASS_ROWS) == 5
    assert all(len(c["others"]) == 1 for c in _CLASS_ROWS)  # others = [최약 F1] 하나


def _class_case(case: dict) -> dict:
    """다중 예시에서 최약 9 의 F1 = others[0], 다른 클래스 5 의 F1 = value 로 둔 사본(판정 대상 지표와 원값이 다르다)."""
    raw = _load("multiclass")
    w = raw["fact_sheet"]["facts"]["weakest_class"]
    w["f1"] = case["others"][0]
    del w["f1_text"]
    for p in raw["fact_sheet"]["per_class"]:
        if p["label"] == "9":
            p["f1"] = case["others"][0]
        if p["label"] == "5":
            p["f1"] = case["value"]
    assert all(abs(m["value"] - case["value"]) > 1e-9 for m in raw["fact_sheet"]["metrics"])
    return raw


@pytest.mark.parametrize("case", _CLASS_ROWS, ids=lambda c: c["expected"])
def test_u2_class_table_rows_are_what_the_backend_class_f1_prints(case):
    """표의 class_table 줄 = 백엔드 클래스별 표 F1(한 함수) = 사실 보기 클래스별 표 = 규칙 문장(Ctx.class_f1)."""
    raw = _class_case(case)
    req = parse_request(copy.deepcopy(raw))[0]
    fs = req.fact_sheet
    assert wording_ed4.class_f1_text_ed4(fs, "5", case["value"]) == case["expected"]
    assert _class_f1(fs, "5", case["value"]) == case["expected"]
    assert _ctx(req, "multiclass").class_f1("5", case["value"]) == case["expected"]
    per_class = {r["클래스"]: r for r in _view(raw)["클래스별 성능"]["클래스별"]}
    assert per_class["5"]["F1"] == case["expected"]


def test_u2_class_table_d2_comes_before_distinct():
    """다른 클래스 F1 이 판정 대상 지표의 원값과 같으면 그 value_text 가 distinct 보다 먼저다(D2)."""
    raw = _load("multiclass")
    m1 = _metric(raw, "M1")
    m1["value_text"] = "0.98889"  # 판정과 어긋나 보여 늘린 표기라고 친다
    for p in raw["fact_sheet"]["per_class"]:
        if p["label"] == "5":
            p["f1"] = m1["value"]
    per_class = {r["클래스"]: r for r in _view(raw)["클래스별 성능"]["클래스별"]}
    assert per_class["5"]["F1"] == "0.98889"


def test_u2_average_gap_reference_class_uses_the_class_table_text_in_the_view_and_the_rule_sentence():
    """다중 평균 차이(8.2-07)의 기준 클래스가 최약 클래스와 셋째 자리로 같아 보이면, 사실 보기와 규칙 문장이 같은 늘린 글자
    (클래스별 표와 같은 한 곳)를 쓴다 — 규칙 문장만 plain(0.972)으로 남지 않는다."""
    raw = _load("multiclass")
    ag = raw["fact_sheet"]["facts"]["average_gap"]
    ag["reference_class"]["label"], ag["reference_class"]["f1"] = "5", 0.9724770642201835  # 최약 9(0.97196)와 0.972 로 같아 보임
    req = parse_request(copy.deepcopy(raw))[0]
    sentence = ri._average_gap(_ctx(req, "multiclass"), "macro_vs_weighted")
    assert "표본이 가장 많은 클래스는 5이며 표본 55건, F1 0.9725이다." in sentence
    gap = _view(raw)["서술용 파생값"]["매크로 평균과 가중 평균의 F1 차이"]
    assert gap["기준 클래스"]["F1"] == "0.9725"


def test_u2_edition3_class_f1_is_unchanged():
    """판 3 은 num(넷째 자리) 그대로 — 같은 입력이라도 판 3 규칙 문장에는 늘린 글자가 없다."""
    raw = as_edition(_load("multiclass"), 3)
    req = parse_request(copy.deepcopy(raw))[0]
    ctx = ri.Ctx(fs=req.fact_sheet, task_type="multiclass", client_inputs={}, derived={}, benchmark_refs=[], edition=3)
    assert ctx.class_f1("5", 0.9724770642201835) == "0.9725"  # 판 3 num 은 넷째 자리
    assert ctx.class_f1("6", 0.9814814814814815) == "0.9815"


# ── U3 D2 보강 — 선언 평균·평균 차이 other_f1 = 판정 대상 지표 원값 ─────────────────────────


def _declared_micro() -> dict:
    """다중 예시: 의뢰자가 '샘플 단위로 중요'를 밝혀 마이크로 평균을 싣고(다중 micro P = R = F1 = Accuracy = M1 원값), M1
    value_text 가 판정 쪽으로 늘린 표기(0.98889)라고 친 사본."""
    raw = _load("multiclass")
    m1 = _metric(raw, "M1")
    m1["value_text"] = "0.98889"
    raw["fact_sheet"]["facts"]["declarations"]["class_priority"] = "samples_first"
    raw["fact_sheet"]["facts"]["declared_average"] = {
        "class_priority": "samples_first", "average": "micro",
        "precision": m1["value"], "recall": m1["value"], "f1": m1["value"],
    }
    return raw


def test_u3_declared_micro_average_f1_is_the_m1_value_text():
    raw = _declared_micro()
    view = _view(raw)["서술용 파생값"]["의뢰자 선언에 맞춘 평균 방식의 결과"]
    assert (view["평균 방식"], view["Precision"], view["Recall"], view["F1"]) == ("마이크로 평균", "0.98889", "0.98889", "0.98889")
    text = _conclusions(raw)["8.2-11"].text
    assert text == "의뢰자가 클래스 중요도를 '샘플 단위로 중요'로 밝혔으므로, 그에 맞춘 마이크로 평균 F1 은 0.98889이다."
    # 확인 숫자(D24) 검사는 그 글자를 받는다 — 규칙 문장을 AI 가 옮겨도 빠짐으로 세지 않는다.
    assert checks.d24_present(text, ri._declared_average_values(_ctx(_prep(raw)[0], "multiclass")), edition=4)
    assert "0.98889" in _prep(raw)[1][2]


def test_u3_declared_average_not_the_same_raw_value_is_plain():
    """같은 원값이 아니면 plain — 늘린 글자가 옮겨 붙지 않는다(이빨: 원값을 1e-6 비틀면 0.989)."""
    raw = _declared_micro()
    raw["fact_sheet"]["facts"]["declared_average"]["f1"] += 1e-6
    view = _view(raw)["서술용 파생값"]["의뢰자 선언에 맞춘 평균 방식의 결과"]
    assert (view["Precision"], view["F1"]) == ("0.98889", "0.989")
    assert _conclusions(raw)["8.2-11"].text.endswith("마이크로 평균 F1 은 0.989이다.")


def test_u3_average_gap_other_f1_same_raw_as_a_judged_metric():
    """평균 차이의 다른 평균 F1(other_f1)이 판정 대상 지표 원값과 같으면 그 value_text — 사실 보기와 8.2-07 규칙 문장이 같다."""
    raw = _load("multiclass")
    m1 = _metric(raw, "M1")
    m1["value_text"] = "0.98889"
    ag = raw["fact_sheet"]["facts"]["average_gap"]
    ag["other_f1"] = m1["value"]
    gap = _view(raw)["서술용 파생값"]["매크로 평균과 가중 평균의 F1 차이"]
    assert gap["가중 평균 F1"] == "0.98889"
    req = parse_request(copy.deepcopy(raw))[0]
    assert "가중 평균 F1 은 0.98889로" in ri._average_gap(_ctx(req, "multiclass"), "macro_vs_weighted")
