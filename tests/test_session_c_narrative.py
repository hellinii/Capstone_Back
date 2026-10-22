"""test_session_c_narrative.py — 서술 라운드 세션 C(품질 확인)에서 고친 것.

실호출 6건 정독(digits·breast cancer·yeast 강·약)과 라운드 검토에서 나온 문제를 장치마다 시험한다.
- 9칸 기대 효과 검사가 숫자 꼴 클래스 이름("클래스 2")을 예측 수치로 오탐하지 않는다(진짜 예측 수치는 계속 잡는다).
- 멀티레이블 잘린 표의 7.1 문장 조사('레이블이 많아').
- 7.2 불균형 말투를 3절과 맞춘다('불균형에 따른 성능 지표 왜곡이 관찰되었다' — 인과 단정·소수 클래스 덜 보기 금지).
- 해석 과장 막기: 오류 성격 일반화, 근소 지표와 신뢰구간, 임계값 조정 여지는 F1 기준, 최약 클래스 동점, 매크로 평균,
  영문 이름 뒤 조사.
- 8.1 모델 값 표기가 참조 범위 위치 판정과 같은 쪽으로 읽힌다.
"""
import pytest

from app.narrative import checks
from app.narrative.schemas import (
    ConfusionCountFact,
    FactSheet,
    MetricFact,
    PerClassFact,
    ProblemFact,
)


def _fs(**kw):
    base = dict(verdict="PASS", score=100.0, metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, status="info")])
    return FactSheet(**{**base, **kw})


# ── 1. 9칸 기대 효과: 숫자 꼴 클래스 이름은 예측 수치가 아니다 ─────────────────────────────

DIGIT_LABELS = frozenset({"0", "1", "2", "8", "10"})


@pytest.mark.parametrize("text, expected", [
    # 클래스 이름 — 뒤에 단위·개선 낱말이 없으면 예측 수치가 아니다(실측 c_digits_weak 9칸이 두 번 걸려 규칙 문장이 됐다).
    ("클래스 2의 재현율이 회복된다.", []),
    ("2(숫자 2) 클래스의 오분류가 줄어든다.", []),
    ("'2' 클래스를 맞히지 못하는 문제가 완화된다.", []),
    ("클래스 2와 8의 누락이 줄어든다.", []),
    ("클래스 10 의 분류가 안정된다.", []),
    # 클래스 이름과 같은 수라도 단위·개선 낱말이 붙으면 예측 수치다.
    ("오분류가 2건 감소한다.", ["2"]),
    ("누락이 2건 줄어든다.", ["2"]),
    ("F1 이 2배 오른다.", ["2"]),
    ("오분류가 2 감소한다.", ["2"]),
    ("재현율이 10%p 오른다.", ["10%"]),
    # 소수·백분율은 이름과 무관하게 예측 수치다.
    ("클래스 2의 F1 이 0.2 향상된다.", ["0.2"]),
    ("클래스 2의 오류가 2% 줄어든다.", ["2%"]),
    # 클래스 이름이 아닌 정수는 지금처럼 잡는다.
    ("클래스 2의 오분류 5건이 사라진다.", ["5"]),
])
def test_class_name_numbers_are_not_predictions(text, expected):
    assert checks.predicted_numbers(text, DIGIT_LABELS) == expected


def test_class_name_numbers_still_caught_without_labels():
    # 이름 목록을 주지 않으면(옛 호출) 지금처럼 잡는다 — 이름 면제는 사실 시트의 이름에서만 나온다.
    assert checks.predicted_numbers("클래스 2의 재현율이 회복된다.") == ["2"]


def test_numeric_labels_come_from_every_name_list():
    fs = _fs(
        per_class=[PerClassFact(label="2", f1=0.0, support=53), PerClassFact(label="cat", f1=0.5, support=10)],
        confusion_counts=[ConfusionCountFact(label="8", tp=0, fp=0, fn=52, tn=300)],
    )
    labels = checks.numeric_labels(fs)
    assert labels == frozenset({"2", "8"})


def test_section_9_check_uses_class_labels():
    fs = _fs(per_class=[PerClassFact(label="2", f1=0.0, support=53)])
    parts = {
        "recommendation_narrative.data_quality": "데이터 권고다.",
        "recommendation_narrative.model_ops": "모델 권고다.",
        "recommendations": [{"problem_id": "P1", "category": "c", "action": "a", "expected_impact": "클래스 2의 누락이 줄어든다."}],
    }
    r = checks.check_section("9", parts, whitelist=set(), verdict="PASS", conclusions=[], problem_ids=["P1"], exempt=set(),
                             labels=checks.numeric_labels(fs))
    assert r.impact_numbers == []
    parts["recommendations"][0]["expected_impact"] = "클래스 2의 누락이 3건 줄어든다."
    r = checks.check_section("9", parts, whitelist=set(), verdict="PASS", conclusions=[], problem_ids=["P1"], exempt=set(),
                             labels=checks.numeric_labels(fs))
    assert r.impact_numbers == ["3"]


async def test_generator_passes_class_labels_to_the_section_9_check(make_section_client):
    # 이진 계약 예시의 클래스는 '0'·'1'. 기대 효과가 '클래스 1' 을 불러도 9칸은 AI 글 그대로(규칙 문장으로 바꾸지 않는다).
    import json
    from pathlib import Path

    from app.narrative import generator
    from app.narrative.schemas import NarrativeRequest

    req = NarrativeRequest.model_validate(json.loads((Path(__file__).parent / "contract" / "fact_sheet_binary.json").read_text()))
    recs = [{"problem_id": i, "category": "점검", "action": "재평가한다.", "expected_impact": "클래스 1의 누락이 줄어든다."}
            for i in ("P1", "P2", "P3")]
    fake = make_section_client({"section_7_1": "의뢰자가 위험하다고 밝힌 FN 은 43건이다.", "recommendations": recs})
    gen = await generator.generate_sections(fake, req)
    sec = gen.trace["sections"]["9"]
    assert sec["checks"][0]["impact_numbers"] == []
    assert gen.section_sources["9"] == "llm"


# ── 2. 조사 — 고정 '가'·'와' 를 받침에 맞춘다 ─────────────────────────────────────────

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from app.narrative.required_items import select_items  # noqa: E402
from app.narrative.schemas import NarrativeRequest  # noqa: E402

CONTRACT = Path(__file__).parent / "contract"


def _raw(name: str) -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text())


def _cons(raw: dict, client_inputs=None) -> dict:
    req = NarrativeRequest.model_validate(raw)
    return {c.item_id: c for c in select_items(req.fact_sheet, client_inputs, req.task_type.value)}


def test_trimmed_multilabel_counts_use_the_right_josa():
    raw = _raw("multilabel")
    raw["fact_sheet"]["facts"]["omitted"] = {"confusion_counts": 3}
    c = _cons(raw)["7.1-05"]
    assert c.text.startswith("레이블이 많아 표에는 일부 레이블만 실렸으며, 실린 레이블 가운데")
    assert "레이블가" not in c.text and c.template_text == c.text


def test_weakest_risk_material_uses_the_right_josa_for_labels():
    raw = _raw("multilabel")
    c = _cons(raw, {"model": {"모델 용도": "기사 분류"}})["8.3-01"]
    assert "그 레이블이 가지는 위험" in c.text and "레이블가" not in c.text


def test_imbalance_conclusion_josa_follows_the_class_name():
    raw = _raw("binary")
    fs = raw["fact_sheet"]
    fs["imbalance_effect"] = {
        "level": "affected", "imbalance_ratio": 1.6719, "accuracy_minus_macro_f1": 0.0011,
        "majority_minus_minority_f1": 0.1092, "majority_class": "benign", "minority_class": "malignant",
    }
    text = _cons(raw)["7.2-02"].text
    assert "다수 클래스 benign과 소수 클래스 malignant의" in text
    text = _cons(raw, {"class_label_descriptions": {"benign": "양성 종양"}})["7.2-02"].text
    assert "다수 클래스 benign(양성 종양)과 소수" in text
    fs["imbalance_effect"]["majority_class"] = "dog"
    assert "다수 클래스 dog와 소수" in _cons(raw)["7.2-02"].text


# ── 3. 7.2 불균형 말투 — 3절과 같은 '불균형에 따른 성능 지표 왜곡이 관찰되었다' ─────────────────

from app.narrative import fact_view, generator, prompt  # noqa: E402
from app.narrative.required_items import imbalance_sentence  # noqa: E402
from app.narrative.schemas import ImbalanceEffectFact  # noqa: E402

CAUSAL = ("영향을 주", "영향을 준", "같은 무게", "덜 중요")


def _affected(raw: dict, *, priority: str | None = None) -> NarrativeRequest:
    fs = raw["fact_sheet"]
    fs["imbalance_effect"] = {
        "level": "affected", "imbalance_ratio": 2.9683, "accuracy_minus_macro_f1": 0.1533,
        "majority_minus_minority_f1": 0.1271, "majority_class": "0", "minority_class": "1",
    }
    if priority is not None:
        fs["facts"]["declarations"]["class_priority"] = priority
    return NarrativeRequest.model_validate(raw)


def test_affected_template_says_distortion_was_observed():
    ie = ImbalanceEffectFact(level="affected", imbalance_ratio=3.2, accuracy_minus_macro_f1=0.12,
                             majority_minus_minority_f1=0.03, majority_class="neg", minority_class="pos")
    text = imbalance_sentence(ie)
    assert text == (
        "불균형 비율(3.2)이 1.5 이상이어서 불균형이 있고, 정확도와 매크로 F1 의 차이(0.12)가 기준(0.1) 이상이어서 "
        "불균형에 따른 성능 지표 왜곡이 관찰되었다. 다수 클래스(neg)와 소수 클래스(pos)의 F1 차이(0.03)는 기준(0.1) 미만이다. "
        "소수 클래스는 표본이 적어 수치가 흔들리기 쉬우므로 성능 지표 해석에 유의가 필요하다."
    )
    ml = ImbalanceEffectFact(level="affected", imbalance_ratio=52.9231, accuracy_minus_macro_f1=None,
                             majority_minus_minority_f1=0.4, majority_class="Class12", minority_class="Class14")
    assert imbalance_sentence(ml).endswith("소수 레이블은 표본이 적어 수치가 흔들리기 쉬우므로 성능 지표 해석에 유의가 필요하다.")
    for t in (text, imbalance_sentence(ml)):
        assert not any(w in t for w in CAUSAL), t


def test_affected_conclusion_and_prompt_note_follow_section_3_wording():
    req = _affected(_raw("binary"))
    cons = {c.item_id: c for c in select_items(req.fact_sheet, None, "binary")}
    assert cons["7.2-02"].text.startswith("불균형에 따른 성능 지표 왜곡이 관찰되었다(비율이 1.5 이상이고 ")
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.2"].notes)
    assert "'불균형에 따른 성능 지표 왜곡이 관찰되었다'로 쓴다" in notes
    assert "영향을 주었다·준 것으로 보인다·판단된다 같은 인과 단정은 쓰지 않는다" in notes
    assert "소수 클래스는 표본이 적어 수치가 흔들리기 쉬우니 해석에 유의해야 한다는 뜻으로 적는다" in notes
    assert "덜 중요하게 보거나 다수 클래스와 같은 무게로 읽지 말라는 말은 쓰지 않는다" in notes
    assert "모든 클래스가 동등" not in notes  # 선언이 없으면 선언을 말하지 않는다
    # 지시문 전체(칸 하는 일·사실 시트 보기)에도 인과 단정 말이 없다.
    msg = prompt.build_user_message(ctx, ("7.1", "7.2", "7.3"))
    for w in ("영향을 주었다고 판단", "영향을 준 것으로", "불균형 영향 있음", "성능 수치에 준 영향"):
        assert w not in msg, w


def test_all_equal_declaration_is_named_in_the_imbalance_note():
    req = _affected(_raw("multilabel"), priority="all_equal")
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.2"].notes)
    assert "의뢰자가 모든 레이블이 동등하다고 밝혔으므로 소수 레이블의 성능도 다른 레이블과 같은 비중으로 다룬다" in notes
    assert "소수 레이블은 표본이 적어" in notes


def test_imbalance_label_in_fact_view_is_observation_wording():
    assert fact_view.IMBALANCE_KO["affected"].startswith("불균형에 따른 성능 지표 왜곡이 관찰됨")


# ── 4a. 오류 성격 — 건수가 적거나 두 오류가 소수면 일반화하지 않는다 ───────────────────────

import re  # noqa: E402

from app.narrative.grounding import build_number_whitelist  # noqa: E402
from app.narrative.required_items import ERROR_NATURE_MIN_COUNT, error_breakdown  # noqa: E402
from app.narrative.schemas import ErrorRatiosFact  # noqa: E402

GENERALIZE_BAN = "'상당 부분'·'대부분'·'우세한 성격'·'~가 중심' 같은 말 없이"


def _er(m, b, c, *, threshold=0.5, unit="sample"):
    return ErrorRatiosFact(
        unit=unit, misclassified=m,
        boundary=None if b is None else {"count": b, "ratio": round(b / m, 4) if m else None},
        boundary_note="threshold_unknown" if b is None else None,
        confident={"count": c, "ratio": round(c / m, 4) if m else None},
        threshold=threshold,
    )


def _nature(er: ErrorRatiosFact, task="binary", *, printed=True):
    raw = _raw(task)
    raw["fact_sheet"]["error_ratios"] = json.loads(er.model_dump_json())
    if not printed:
        raw["fact_sheet"].pop("facts", None)
    req = NarrativeRequest.model_validate(raw)
    return next(c for c in select_items(req.fact_sheet, None, task) if c.item_id == "7.1-11")


def test_error_breakdown_counts_neither_only_when_the_two_kinds_cannot_overlap():
    assert error_breakdown(_er(3, 1, 0)) == (1, 0, 2)                         # 이진 임계값 0.5 — 겹칠 수 없다
    assert error_breakdown(_er(6, 2, 0, threshold=None)) == (2, 0, 4)          # 다중(최댓값 선택) — 겹칠 수 없다
    assert error_breakdown(_er(10, 3, 4, threshold=0.85)) == (3, 4, None)      # 임계값 0.85 — 0.9 이상 확률이 경계 안에도 든다
    assert error_breakdown(_er(10, 3, 4, threshold=0.15)) == (3, 4, None)
    assert error_breakdown(_er(10, 3, 4, threshold={"a": 0.5, "b": 0.85})) == (3, 4, None)
    assert error_breakdown(_er(10, 3, 4, threshold=0.8)) == (3, 4, 3)          # 경계 끝 0.9 는 '미만'이라 겹치지 않는다
    assert error_breakdown(_er(49, None, 1)) is None
    assert error_breakdown(_er(0, 0, 0)) is None


def test_few_boundary_errors_are_not_generalized():
    # 실측 bc_strong: 오분류 3건 중 경계 1건·확신 0건 — '오류의 상당 부분은 경계 근처'라고 썼다.
    item = _nature(_er(3, 1, 0))
    assert item.text.startswith("경계 오류가 확신 오류보다 많다.")
    assert "우세한 오류는" not in item.text
    assert "근거 건수(해석의 근거로만 보고 문장에 쓰지 않는다): 확률이 있는 오분류 3건 가운데 경계 오류 1건, 확신 오류 0건, 둘 다 아닌 오류 2건." in item.text
    assert "오분류 건수가 적고 경계 오류와 확신 오류를 합쳐도 오분류의 절반에 이르지 않아" in item.text
    assert GENERALIZE_BAN in item.text and "'경계 오류가 확신 오류보다 많다' 수준으로만 쓴다" in item.text
    # 규칙 템플릿(성적서가 17 을 찍음)은 숫자 없이 그대로 — 정도의 주장이 없다.
    assert item.template_text.startswith("확률이 있는 오분류 가운데 경계 오류가 확신 오류보다 많다.")
    assert not re.search(r"\d", item.template_text)


def test_minority_without_few_count_and_few_count_without_minority():
    # 오분류는 충분하지만 두 오류가 소수(digits 모양을 크게): 반 미만이라는 사유만.
    item = _nature(_er(40, 12, 3, threshold=None), "multiclass")
    assert "경계 오류와 확신 오류를 합쳐도 오분류의 절반에 이르지 않아" in item.text and "건수가 적" not in item.text
    assert "둘 다 아닌 오류 25건" in item.text
    # 과반이지만 건수가 적다(기준 바로 아래).
    item = _nature(_er(ERROR_NATURE_MIN_COUNT - 1, 7, 0))
    assert "오분류 건수가 적어" in item.text and "절반" not in item.text and GENERALIZE_BAN in item.text
    # 기준과 같으면(과반·충분) 일반화를 막지 않고 우세한 오류를 말한다.
    item = _nature(_er(ERROR_NATURE_MIN_COUNT, 7, 0))
    assert GENERALIZE_BAN not in item.text and "(우세한 오류는 경계 오류)" in item.text
    # 두 오류가 겹칠 수 있으면 둘 다 아닌 건수를 싣지 않고, 큰 쪽만으로 과반을 본다.
    item = _nature(_er(20, 6, 7, threshold=0.85))
    assert "둘 다 아닌" not in item.text and "절반에 이르지 않아" in item.text


def test_a_few_confident_errors_are_not_generalized():
    # 실측 bc_weak: 확신 오류 49건 중 1건 — '확률이 높은 예측 전반을 믿을 수 없다'고 썼다.
    item = _nature(_er(49, None, 1))
    assert "확신 오류는 오분류 가운데 소수다. 확률이 높은 예측 전반을 믿을 수 없다고 일반화하지 말고" in item.text
    item = _nature(_er(40, 2, 1))
    assert "확신 오류는 오분류 가운데 소수다." in item.text
    item = _nature(_er(40, 3, 30))
    assert "소수다" not in item.text


def test_neither_count_is_allowed_and_told_not_to_be_written():
    raw = _raw("multiclass")  # 계약 예시: 오분류 133건, 경계 27건, 확신 2건 → 둘 다 아닌 104건
    req = NarrativeRequest.model_validate(raw)
    _, _, whitelist, _, ctx, _, _ = generator._prepare(req)
    assert "104" in whitelist
    assert "104" not in build_number_whitelist(req.fact_sheet, [], {})  # 원래 목록에는 없다 — 생성기가 넣는다
    msg = prompt.build_user_message(ctx, ("7.1", "7.2", "7.3"))
    assert "둘 다 아닌 오류 104건" in msg
    assert "해석 재료의 건수도 문장에 쓰지 않는다" in msg


def test_old_requests_are_told_not_to_write_the_neither_count():
    item = _nature(_er(3, 1, 0), printed=False)
    assert "근거 건수(둘 다 아닌 오류의 건수는 사실 시트에 없는 값이라 문장에 쓰지 않는다)" in item.text
