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
                             labels=checks.numeric_labels(fs), edition=3)
    assert r.impact_numbers == []
    parts["recommendations"][0]["expected_impact"] = "클래스 2의 누락이 3건 줄어든다."
    r = checks.check_section("9", parts, whitelist=set(), verdict="PASS", conclusions=[], problem_ids=["P1"], exempt=set(),
                             labels=checks.numeric_labels(fs), edition=3)
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
    return {c.item_id: c for c in select_items(req.fact_sheet, client_inputs, req.task_type.value, edition=3)}


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


def _affected(raw: dict, *, priority: str | None = None, minority_small: bool = False) -> NarrativeRequest:
    fs = raw["fact_sheet"]
    fs["imbalance_effect"] = {
        "level": "affected", "imbalance_ratio": 2.9683, "accuracy_minus_macro_f1": 0.1533,
        "majority_minus_minority_f1": 0.1271, "majority_class": "0", "minority_class": "1",
    }
    # 소수 쪽이 작은 표본인지는 facts.low_support_classes 가 정한다(세션 C 재확인 — 64건 클래스에도 '표본이 적어'가 붙었다).
    fs["facts"]["low_support_classes"] = [
        r for r in fs["facts"].get("low_support_classes", []) if r["label"] != "1"
    ] + ([{"label": "1", "support": 12, "share": 0.04, "share_text": "0.04"}] if minority_small else [])
    if priority is not None:
        fs["facts"]["declarations"]["class_priority"] = priority
    return NarrativeRequest.model_validate(raw)


def test_affected_template_says_distortion_was_observed():
    ie = ImbalanceEffectFact(level="affected", imbalance_ratio=3.2, accuracy_minus_macro_f1=0.12,
                             majority_minus_minority_f1=0.03, majority_class="neg", minority_class="pos")
    head = (
        "불균형 비율(3.2)이 1.5 이상이어서 불균형이 있고, 정확도와 매크로 F1 의 차이(0.12)가 기준(0.1) 이상이어서 "
        "불균형에 따른 성능 지표 왜곡이 관찰되었다. 다수 클래스(neg)와 소수 클래스(pos)의 F1 차이(0.03)는 기준(0.1) 미만이다. "
    )
    # 소수 쪽이 작은 표본일 때만 '표본이 적어 흔들리기 쉽다'. 아니면 해석에 유의하라는 말만.
    text = imbalance_sentence(ie, minority_small=True)
    assert text == head + "소수 클래스는 표본이 적어 수치가 흔들리기 쉬우므로 성능 지표 해석에 유의가 필요하다."
    plain = imbalance_sentence(ie)
    assert plain == head + "소수 클래스의 성능 지표 해석에 유의가 필요하다."
    ml = ImbalanceEffectFact(level="affected", imbalance_ratio=52.9231, accuracy_minus_macro_f1=None,
                             majority_minus_minority_f1=0.4, majority_class="Class12", minority_class="Class14")
    assert imbalance_sentence(ml, minority_small=True).endswith("소수 레이블은 표본이 적어 수치가 흔들리기 쉬우므로 성능 지표 해석에 유의가 필요하다.")
    assert imbalance_sentence(ml).endswith("소수 레이블의 성능 지표 해석에 유의가 필요하다.")
    for t in (text, plain, imbalance_sentence(ml)):
        assert not any(w in t for w in CAUSAL), t


def test_affected_conclusion_and_prompt_note_follow_section_3_wording():
    req = _affected(_raw("binary"))
    cons = {c.item_id: c for c in select_items(req.fact_sheet, None, "binary", edition=3)}
    assert cons["7.2-02"].text.startswith("불균형에 따른 성능 지표 왜곡이 관찰되었다(비율이 1.5 이상이고 ")
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.2"].notes)
    assert "'불균형에 따른 성능 지표 왜곡이 관찰되었다'로 쓴다" in notes
    assert "영향을 주었다·준 것으로 보인다·판단된다 같은 인과 단정은 쓰지 않는다" in notes
    # 소수 클래스 '1' 이 작은 표본이 아니면 '표본이 적어 흔들린다'고 쓰지 못하게 한다.
    assert "소수 클래스 1은 표본이 적은 클래스가 아니므로 표본이 적어서 수치가 흔들린다고 쓰지 않는다" in notes
    assert "표본이 적어 수치가 흔들리기 쉬우니" not in notes
    assert "덜 중요하게 보거나 다수 클래스와 같은 무게로 읽지 말라는 말은 쓰지 않는다" in notes
    assert "모든 클래스가 동등" not in notes  # 선언이 없으면 선언을 말하지 않는다
    # 지시문 전체(칸 하는 일·사실 시트 보기)에도 인과 단정 말이 없다.
    msg = prompt.build_user_message(ctx, ("7.1", "7.2", "7.3"))
    for w in ("영향을 주었다고 판단", "영향을 준 것으로", "불균형 영향 있음", "성능 수치에 준 영향"):
        assert w not in msg, w


def test_small_minority_gets_the_small_sample_note_and_template():
    req = _affected(_raw("binary"), minority_small=True)
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.2"].notes)
    assert "소수 클래스는 표본이 적어 수치가 흔들리기 쉬우니 해석에 유의해야 한다는 뜻으로 적는다" in notes
    assert "표본이 적은 클래스가 아니므로" not in notes
    from app.narrative import fallback
    def template(r):
        derived, refs, _, conclusions, _, _, _ = generator._prepare(r)
        return fallback.section_template("7.2", r.fact_sheet, refs, derived, task_type="binary", conclusions=conclusions, edition=3)

    big = template(_affected(_raw("binary")))
    small = template(req)
    text_of = lambda parts: " ".join(v for v in parts.values() if isinstance(v, str))  # noqa: E731
    assert "표본이 적어 수치가 흔들리기 쉬우므로" in text_of(small)
    assert "표본이 적어 수치가 흔들리기 쉬우므로" not in text_of(big)
    assert "소수 클래스의 성능 지표 해석에 유의가 필요하다." in text_of(big)


def test_all_equal_declaration_is_named_in_the_imbalance_note():
    req = _affected(_raw("multilabel"), priority="all_equal")
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.2"].notes)
    assert "의뢰자가 모든 레이블이 동등하다고 밝혔으므로 소수 레이블의 성능도 다른 레이블과 같은 비중으로 다룬다" in notes
    assert "표본이 적은 레이블이 아니므로" in notes
    # 조사는 이름 끝소리로 — '다수 레이블와'가 아니다(실호출에서 AI 가 지시문 문장을 그대로 옮겨 적은 적이 있다).
    assert "덜 중요하게 보거나 다수 레이블과 같은 무게로 읽지 말라는 말은 쓰지 않는다" in notes
    assert "레이블와" not in notes


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
    return next(c for c in select_items(req.fact_sheet, None, task, edition=3) if c.item_id == "7.1-11")


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
    # 이때는 주된 성격으로 해석해도 된다고 적는다(세션 C 재확인 — digits_weak 249건 중 경계 171건인데 AI 가 해석을 보류).
    assert "건수가 충분하고 경계 오류가 오분류의 절반을 넘으므로 오류의 주된 성격으로 해석해도 된다." in item.text
    assert "주된 성격으로 해석해도" not in _nature(_er(3, 1, 0)).text
    # 두 오류가 겹칠 수 있으면 둘 다 아닌 건수를 싣지 않고, 큰 쪽만으로 과반을 본다.
    item = _nature(_er(20, 6, 7, threshold=0.85))
    # 합은 부풀 수 있어 '합쳐도 절반에 이르지 않는다'고 말하지 않는다(세션 C 마지막 검토).
    assert "둘 다 아닌" not in item.text and "많은 쪽 오류도 오분류의 절반을 넘지 않아" in item.text


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
    assert "104" not in build_number_whitelist(req.fact_sheet, [], {}, edition=3)  # 원래 목록에는 없다 — 생성기가 넣는다
    msg = prompt.build_user_message(ctx, ("7.1", "7.2", "7.3"))
    assert "둘 다 아닌 오류 104건" in msg
    assert "해석 재료의 건수도 문장에 쓰지 않는다" in msg


def test_old_requests_are_told_not_to_write_the_neither_count():
    item = _nature(_er(3, 1, 0), printed=False)
    assert "근거 건수(둘 다 아닌 오류의 건수는 사실 시트에 없는 값이라 문장에 쓰지 않는다)" in item.text


# ── 4b. 근소 지표 — 신뢰구간이 목표를 포함하지 않으면 '판정이 뒤집힐 수 있다'를 쓰지 않는다 ───────────

from app.narrative.grounding import find_verdict_contradictions, verify_grounding  # noqa: E402

FLIP = "표본이 달라지면"


def _near(name, mid, status, contains, *, value=None):
    v = value if value is not None else (0.9844 if status == "pass" else 0.9756)
    return MetricFact(metric_id=mid, display_name=name, value=v, threshold=0.98, status=status, near_target=True,
                      margin=round(v - 0.98, 4), margin_text=f"{round(v - 0.98, 4):g}", value_text=f"{v:g}",
                      ci_low=0.95 if contains else 0.9973, ci_high=0.9999 if contains else 1.0,
                      ci_contains_target=contains)


def _near_sheet(metrics, *, printed=True, verdict="PASS"):
    facts = {"fixed_sentences": [{"id": "18", "section": "8.3", "text": "근소 차이 문장"}]} if printed else None
    return FactSheet(verdict=verdict, score=100.0, metrics=metrics, facts=facts)


def _item(fs, item_id="8.3-02"):
    return next(c for c in select_items(fs, None, "binary", edition=3) if c.item_id == item_id)


def test_near_material_says_which_intervals_contain_the_target():
    # 실측 bc_strong: Recall·F1 은 구간이 목표 포함, AUROC 는 포함하지 않는데 8.3 이 셋 다 '뒤집힐 수 있다'고 썼다.
    fs = _near_sheet([_near("Recall", "M3", "pass", True), _near("F1 Score", "M4", "pass", True), _near("AUROC", "M7", "pass", False)])
    text = _item(fs).text
    assert text.startswith("근소 차이 지표 가운데 목표를 충족한 지표는 Recall, F1 Score, AUROC이다.")
    assert "신뢰구간이 목표값을 포함하는 지표는 Recall, F1 Score이고, 신뢰구간이 목표값을 포함하지 않는 지표는 AUROC이다." in text
    assert "포함하지 않는 지표는 표본이 달라지면 판정이 뒤집힐 수 있다고 쓰지 말고 목표와의 여유가 작다는 사실만 쓴다." in text
    assert "근소 차이 지표는 이 지표들뿐이다. 다른 지표를 여유가 작다거나 판정이 뒤집힐 수 있다고 해석하지 않는다." in text


def test_near_template_splits_by_interval():
    fs = _near_sheet([_near("Recall", "M3", "pass", True), _near("AUROC", "M7", "pass", False)])
    t = _item(fs).template_text
    assert t == (
        "목표값에 근소한 차이로 충족한 지표(Recall)는 표본이 달라지면 미달로 바뀔 수 있어 판정의 여유가 작다. "
        "근소 차이 지표 가운데 신뢰구간이 목표값을 포함하지 않는 지표(AUROC)는 목표와의 여유가 작다."
    )
    # 4b: 포함하지 않는 지표는 '여유가 작다는 사실만' — 판정 안정성에 대한 새 주장(가능성은 낮다)을 덧붙이지 않는다.
    assert "가능성" not in t
    only_safe = _near_sheet([_near("AUROC", "M7", "pass", False)])
    t = _item(only_safe).template_text
    assert FLIP + "면" not in t and "바뀔 수 있어" not in t and "AUROC" in t
    # 구간이 모두 목표를 포함하면(또는 구간이 없으면) 종전 문장 그대로.
    assert _item(_near_sheet([_near("Recall", "M3", "pass", True)])).template_text == (
        "목표값에 근소한 차이로 충족한 지표는 표본이 달라지면 미달로 바뀔 수 있어 판정의 여유가 작다."
    )


def test_near_template_for_old_requests_splits_by_interval():
    fs = _near_sheet([_near("Recall", "M3", "pass", True), _near("AUROC", "M7", "pass", False)], printed=False)
    t = _item(fs).template_text
    assert t.startswith("목표값과의 차이가 목표값의 3% 미만인 아슬아슬 합격 지표는 Recall(0.9844, 목표 0.98, 차이 0.0044)로, "
                        "표본이 달라지면 목표에 미달할 위험이 있다.")
    assert ("아슬아슬 합격 지표 가운데 AUROC(0.9844, 목표 0.98, 차이 0.0044)는 신뢰구간이 목표값을 포함하지 않으며, "
            "목표와의 여유가 작다.") in t
    assert "가능성" not in t


@pytest.mark.parametrize("printed", [True, False])
@pytest.mark.parametrize("verdict, statuses", [("PASS", ("pass", "pass")), ("FAIL", ("fail", "pass")), ("FAIL", ("fail", "fail"))])
def test_near_templates_pass_their_own_checks(printed, verdict, statuses):
    metrics = [_near("Recall", "M3", statuses[0], True), _near("AUROC", "M7", statuses[1], False)]
    fs = _near_sheet(metrics, printed=printed, verdict=verdict)
    t = _item(fs).template_text
    if t is None:  # 옛 요청은 합격한 근소 지표만 쓴다
        assert not printed and all(s == "fail" for s in statuses)
        return
    assert verify_grounding([t], build_number_whitelist(fs, [], {}, edition=3)).passed, t
    assert find_verdict_contradictions([t], verdict) == [], t


def test_prompt_limits_near_interpretation_to_near_metrics():
    req = NarrativeRequest.model_validate(_raw("binary"))
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    msg = prompt.build_user_message(ctx, ("8.1", "8.2", "8.3"))
    assert ("표본이 달라지면 판정이 뒤집힐 수 있다는 말은 신뢰구간이 목표값을 포함하는 지표나 구간이 없는 지표에만 쓰고, "
            "포함하지 않는 지표는 목표와의 여유가 작다는 사실만 쓴다.") in msg
    # 근소 지표가 없어도(실측 yeast_strong 의 Hamming Loss) 8.3 노트가 근소 해석을 근소 지표로 한정한다.
    raw = _raw("multilabel")
    req = NarrativeRequest.model_validate(raw)
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["8.3"].notes)
    assert "목표와의 여유가 작다·아슬아슬하다·판정이 뒤집힐 수 있다는 해석은 사실 시트 지표 표의 '근소 차이' 칸이 '예'인 지표에만 쓴다" in notes


# ── 4c. 임계값 조정 여지는 F1 기준 판단이다 ─────────────────────────────────────────

F1_BASIS = ("임계값 조정 여지는 F1 을 가장 높이는 결정 임계값에서 F1 이 얼마나 오르는지만 본 판단이다. "
            "Recall·Precision 같은 다른 지표의 문제에 임계값 조정이 소용없다거나 임계값으로 해소할 수 없다고 적용하지 않는다.")


def _gain_raw(notable: bool) -> dict:
    raw = _raw("binary")
    g = raw["fact_sheet"]["f1_threshold_gain"]
    if not notable:
        g.update(max_f1=0.9844, current_f1=0.9767, gain=0.0077, gain_text="0.0077", best_threshold=0.6166, notable=False)
        raw["fact_sheet"]["problems"] = [p for p in raw["fact_sheet"]["problems"] if p["kind"] != "threshold_tuning"]
    return raw


@pytest.mark.parametrize("notable", [True, False])
def test_threshold_gain_is_f1_only_in_every_section_that_may_mention_it(notable):
    req = NarrativeRequest.model_validate(_gain_raw(notable))
    assert req.fact_sheet.f1_threshold_gain is not None
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    for sid in ("7.3", "8.2", "8.3", "9"):
        assert F1_BASIS in ctx.briefs[sid].notes, (sid, ctx.briefs[sid].notes)
    view = ctx.view["임계값 조정 여지"]
    assert view["판단 기준"] == "F1 만 본 판단(Recall·Precision 등 다른 지표에 대한 판단이 아님)"
    risk_note = "평가 엔진은 임계값 조정 여지가 크지 않다고 봤다. 결정 임계값을 위험으로 쓰지 않는다."
    assert (risk_note in ctx.briefs["8.3"].notes) is (not notable)


def test_no_threshold_note_without_gain():
    req = NarrativeRequest.model_validate(_raw("multiclass"))
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    assert all(F1_BASIS not in ctx.briefs[sid].notes for sid in ("7.3", "8.2", "8.3", "9"))


# ── 4d. 최약 클래스 동점 — 하나만 단독 최약처럼 쓰지 않는다 ─────────────────────────────

from app.narrative.required_items import weakest_ties  # noqa: E402


def _tie_raw(task="multiclass", *, bird_f1=0.27421, trimmed=False, extra=0):
    """계약 예시(최약 cat F1 0.2742)에서 bird 의 F1 을 넷째 자리로 같게 — 판정 자릿수 기준 동점."""
    raw = _raw(task)
    fs = raw["fact_sheet"]
    for p in fs["per_class"]:
        if p["label"] == "bird":
            p["f1"] = bird_f1
    for i in range(extra):  # 동점 클래스를 더 붙인다(숫자 꼴 이름 — Digits 처럼)
        fs["per_class"].append({"label": str(i + 2), "precision": 0.0, "recall": 0.0, "f1": 0.27419, "support": 40 + i})
    if trimmed:
        fs["facts"]["omitted"] = {"per_class": 7}
    return raw


def test_weakest_ties_by_printed_digits():
    req = NarrativeRequest.model_validate(_tie_raw())
    assert weakest_ties(req.fact_sheet) == (["cat", "bird"], False)
    req = NarrativeRequest.model_validate(_tie_raw(bird_f1=0.2743))
    assert weakest_ties(req.fact_sheet) == ([], False)


def test_tied_weakest_conclusion_names_all_of_them():
    cons = _cons(_tie_raw(), {"class_label_descriptions": {"cat": "고양이", "bird": "새"}})
    text = cons["7.1-10"].text
    assert text.startswith(
        "F1 이 가장 낮은 클래스는 F1 0.2742로 같은 두 클래스(cat(고양이), bird(새))이며, "
        "그 가운데 cat(고양이)의 표본은 57건으로 전체 표본의 28.5%이다."
    )
    assert cons["7.1-10"].template_text == text
    assert cons["8.3-01"].template_text == "F1 이 가장 낮은 클래스는 F1 0.2742로 같은 두 클래스(cat(고양이), bird(새))이다."


def test_many_ties_list_a_few_names_and_trimmed_tables_limit_the_claim():
    cons = _cons(_tie_raw(extra=9))
    text = cons["7.1-10"].text
    assert text.startswith("F1 이 가장 낮은 클래스는 F1 0.2742로 같은 여러 클래스(cat, bird, 2, 3, 4 등)이며,")
    cons = _cons(_tie_raw(extra=4))
    assert cons["7.1-10"].text.startswith("F1 이 가장 낮은 클래스는 F1 0.2742로 같은 여섯 클래스(cat, bird, 2, 3, 4 등)이며,")
    cons = _cons(_tie_raw(trimmed=True))
    assert cons["7.1-10"].text.startswith("표에 실린 클래스 가운데 F1 이 가장 낮은 클래스는 F1 0.2742로 같은 두 클래스(cat, bird)이며,")
    assert cons["8.3-01"].template_text.startswith("표에 실린 클래스 가운데 F1 이 가장 낮은 클래스는")


def test_tied_weakest_notes_and_self_check():
    req = NarrativeRequest.model_validate(_tie_raw(extra=3))
    _, _, whitelist, conclusions, ctx, exempt, _ = generator._prepare(req)
    for sid in ("7.1", "8.3"):
        notes = " ".join(ctx.briefs[sid].notes)
        assert "F1 이 같은 최하위 클래스가 여럿이다(cat, bird, 2, 3, 4). 한 클래스만 가장 약한 클래스처럼 쓰지 말고 함께 밝힌다." in notes, sid
    for c in conclusions:
        if c.item_id in ("7.1-10", "8.3-01"):
            assert verify_grounding([c.template_text], whitelist).passed, c.template_text
            assert checks.code_words([c.template_text], exempt) == []


def test_no_tie_keeps_the_single_weakest_sentence():
    cons = _cons(_raw("multiclass"))
    assert cons["7.1-10"].text.startswith("F1 이 가장 낮은 클래스는 F1 0.2742인 cat이며, 표본은 57건으로")
    assert cons["8.3-01"].template_text == "F1 이 가장 낮은 클래스는 F1 0.2742인 cat이다."


def test_tied_weakest_labels_in_multilabel_use_label_josa_and_view():
    raw = _raw("multilabel")  # 최약 sports F1 0.4322
    for p in raw["fact_sheet"]["per_class"]:
        if p["label"] == "finance":
            p["f1"] = 0.43218
    req = NarrativeRequest.model_validate(raw)
    _, _, _, conclusions, ctx, _, _ = generator._prepare(req)
    notes = " ".join(ctx.briefs["7.1"].notes)
    assert "F1 이 같은 최하위 레이블이 여럿이다(sports, finance). 한 레이블만 가장 약한 레이블처럼 쓰지 말고 함께 밝힌다." in notes
    assert ctx.view["서술용 파생값"]["F1 이 가장 낮은 레이블"]["F1 이 같은 레이블"] == ["sports", "finance"]
    risk = next(c for c in conclusions if c.item_id == "8.3-01")
    assert risk.template_text == "F1 이 가장 낮은 레이블은 F1 0.4322로 같은 두 레이블(sports, finance)이다."


# ── 4e. 매크로 평균 — 평균 방식에 맞게만 설명한다 ─────────────────────────────────────


def test_metric_rows_carry_the_averaging_method():
    # 엔진(app/evaluation/metrics/common.py): 이진은 양성 클래스 기준, 다중·멀티레이블의 Precision·Recall·F1·F-beta 는 매크로 평균.
    for task, noun in (("multiclass", "클래스"), ("multilabel", "레이블")):
        _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw(task)))
        f1 = next(r for r in ctx.view["지표"] if r["지표"] == "F1 Score")
        assert f1["평균 방식"] == f"매크로 평균({noun}별 값의 단순 평균 — {noun}마다 같은 무게)"
    _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw("multilabel")))
    jac = next(r for r in ctx.view["지표"] if r["지표"] == "Jaccard Index")
    assert jac["평균 방식"] == "표본 평균(표본마다 값을 구해 평균)"
    hamming = next(r for r in ctx.view["지표"] if r["지표"] == "Hamming Loss")
    assert "평균 방식" not in hamming
    _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw("binary")))
    f1 = next(r for r in ctx.view["지표"] if r["지표"] == "F1 Score")
    assert f1["평균 방식"] == "양성 클래스 기준(평균 아님)"


MACRO_NOTE = ("F1 Score 값은 레이블별 값의 매크로 평균(단순 평균)이다. 레이블마다 같은 무게라 약한 레이블의 부진이 이 평균에 그대로 "
              "반영된다. '평균 지표에는 약한 레이블의 부진이 드러나지 않을 수 있다'처럼 가중 평균·마이크로 평균에만 맞는 설명을 "
              "이 지표에 쓰지 않는다. 평균 방식은 사실 시트 지표 표의 '평균 방식'에 맞게만 설명한다.")


def test_macro_note_in_sections_that_interpret_weak_classes():
    _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw("multilabel")))
    for sid in ("7.1", "8.2", "8.3"):
        assert MACRO_NOTE in ctx.briefs[sid].notes, sid
    _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw("multiclass")))
    assert any(n.startswith("Precision, Recall, F1 Score 값은 클래스별 값의 매크로 평균(단순 평균)이다.") for n in ctx.briefs["8.3"].notes)
    _, _, _, _, ctx, _, _ = generator._prepare(NarrativeRequest.model_validate(_raw("binary")))
    assert not any("매크로 평균(단순 평균)" in n for sid in ("7.1", "8.2", "8.3") for n in ctx.briefs[sid].notes)


# ── 4f. 영문·숫자 이름 뒤 조사 ─────────────────────────────────────────────────────


def test_system_prompt_has_the_josa_rule_for_latin_names():
    # 실호출: 'benign(양성 종양)와'·'benign가'·'P3가' 가 나왔다.
    assert ("영문·숫자 이름 뒤 조사는 그 이름을 읽는 소리의 끝소리에 맞춘다. 영문 이름은 영어 발음의 끝소리로 고른다"
            "(benign과, cat과, dog와, P3이).") in prompt.SYSTEM_PROMPT


# ── 5. 8.1 모델 값 표기가 참조 범위 위치 판정과 같은 쪽으로 읽힌다 ─────────────────────────────

from app.narrative.baselines import build_benchmark_refs  # noqa: E402
from app.narrative.derived import compute_derived  # noqa: E402
from app.narrative.wording import range_value_text  # noqa: E402


@pytest.mark.parametrize("value, low, high, base, expected", [
    (0.79996, 0.8, 0.92, None, "0.79996"),      # 넷째 자리 '0.8' 은 범위 안으로 읽힌다 — 미흡 판정
    (0.9500004, 0.85, 0.95, None, "0.950001"),   # '0.95' 는 범위 끝 — 우수 판정. 여섯째 자리로도 같아 위치 쪽(올림)으로 자른다
    (0.8497, 0.85, 0.92, None, "0.8497"),
    (0.92049, 0.85, 0.92, None, "0.9205"),
    (0.9, 0.85, 0.92, None, "0.9"),
    (0.85, 0.85, 0.92, None, "0.85"),            # 범위 끝과 같으면 범위 안
    (0.84996, 0.85, 0.92, "0.85", "0.84996"),    # 판정 대상의 6.1 표기(셋째 자리)도 범위 위치와 어긋나면 늘린다
    (0.861, 0.85, 0.92, "0.861", "0.861"),       # 어긋나지 않으면 6.1 표기 그대로
    (0.7999999999999, 0.8, 0.92, None, "0.799999"),  # 여섯째 자리로도 같아 보이면 판정 쪽으로 자른다(프론트와 같은 최대 자릿수)
])
def test_range_value_text(value, low, high, base, expected):
    assert range_value_text(value, low, high, base) == expected


def _bench_fs():
    return FactSheet(verdict="PASS", score=100.0, metrics=[
        MetricFact(metric_id="M2", display_name="Precision", value=0.79996, status="info"),
        MetricFact(metric_id="M7", display_name="AUROC", value=0.9500004, status="info"),
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.84996, threshold=0.8, status="pass", value_text="0.85",
                   margin=0.04996, margin_text="0.05"),
    ])


def test_benchmark_rule_sentence_and_fact_view_use_position_consistent_values():
    fs = _bench_fs()
    refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    text = next(c.template_text for c in select_items(fs, None, "binary", benchmark_refs=refs, edition=3) if c.item_id == "8.1-01")
    assert "참조 범위보다 우수한 지표는 AUROC(0.950001, 참조 범위 0.85~0.95)이고" in text
    assert "Precision(0.79996, 참조 범위 0.8~0.92)" in text and "Accuracy(0.84996, 참조 범위 0.85~0.92)" in text
    derived = compute_derived(fs, task_type="binary")
    view = fact_view.build_fact_view(fs, derived, refs, task_type="binary", task_label="이진 분류", report_purpose="internal", edition=3)
    rows = {r["지표"]: r for r in view["벤치마크(내부 참조 기준)"]["지표별"]}
    assert (rows["Precision"]["모델 값"], rows["Precision"]["평가(방향 반영)"]) == ("0.79996", "미흡")
    assert (rows["AUROC"]["모델 값"], rows["AUROC"]["평가(방향 반영)"]) == ("0.950001", "우수")
    assert rows["Accuracy"]["모델 값"] == "0.84996"
    # 규칙 문장이 자기 검문(숫자 허용 목록)을 통과한다.
    assert verify_grounding([text], build_number_whitelist(fs, refs, derived, edition=3)).passed, text
    # 응답의 벤치마크 표(계약)는 그대로 — model_value 는 넷째 자리 숫자다.
    assert [r["model_value"] for r in refs] == [0.8, 0.95, 0.85]


# ── 세션 C 의 새 문장이 모두 들어간 사실 시트에서도 규칙 템플릿이 자기 검문을 통과한다 ─────────────────

from app.narrative import fallback  # noqa: E402
from app.narrative.sections import SECTION_IDS  # noqa: E402


@pytest.mark.parametrize("verdict", ["PASS", "CONDITIONAL_PASS", "FAIL", "INCONCLUSIVE"])
@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_session_c_templates_pass_their_own_checks(task, verdict):
    from test_s6_templates import _request  # 판정 4가지로 바꾼 계약 예시

    req = _request(task, verdict)
    raw = json.loads(req.model_dump_json())
    fs = raw["fact_sheet"]
    # 불균형 영향 있음(affected) — 새 7.2 말투.
    fs["imbalance_effect"] = {
        "level": "affected", "imbalance_ratio": 2.9683,
        "accuracy_minus_macro_f1": None if task == "multilabel" else 0.1533,
        "majority_minus_minority_f1": 0.1271, "majority_class": fs["per_class"][0]["label"],
        "minority_class": fs["per_class"][1]["label"],
    }
    # 최약 클래스 동점.
    w = fs["facts"]["weakest_class"]
    other = next(p for p in fs["per_class"] if p["label"] != w["label"])
    other["f1"] = w["f1"]
    # 근소 지표 가운데 구간이 목표를 포함하지 않는 것(FAIL 예시는 근소 지표가 있다).
    for m in fs["metrics"]:
        if m.get("near_target"):
            m["ci_contains_target"] = False
            m["ci_low"], m["ci_high"] = (m["threshold"] + 0.0011, m["threshold"] + 0.0371) if m["status"] == "pass" else (
                m["threshold"] - 0.0371, m["threshold"] - 0.0011)
            break
    req = NarrativeRequest.model_validate(raw)
    fs_obj, ci = req.fact_sheet, req.client_inputs
    derived = compute_derived(fs_obj, task_type=task)
    refs = build_benchmark_refs(task, fs_obj.metrics, edition=3)
    whitelist = build_number_whitelist(fs_obj, refs, derived, ci, edition=3)
    conclusions = select_items(fs_obj, ci, task, edition=3)
    exempt = checks.name_exemptions(fs_obj, ci)
    texts = []
    for sid in SECTION_IDS:
        parts = generator._rec_dicts(fallback.section_template(sid, fs_obj, refs, derived, task_type=task, client_inputs=ci,
                                                              conclusions=conclusions, edition=3))
        if sid == "9" and not fs_obj.problems:
            continue
        r = checks.check_section(sid, parts, whitelist=whitelist, verdict=fs_obj.verdict, conclusions=conclusions,
                                 problem_ids=[p.id for p in fs_obj.problems], exempt=exempt,
                                 masks=checks.verdict_masks(fs_obj, ci), labels=checks.numeric_labels(fs_obj), edition=3)
        assert r.hard_failures == [], (sid, r.as_record(), parts)
        texts += [t for t in checks.section_texts(sid, parts) if t]
    joined = " ".join(texts)
    assert "불균형에 따른 성능 지표 왜곡이 관찰되었다" in joined and "같은 두" in joined
    if (task, verdict) == ("binary", "FAIL"):  # 근소 지표는 이진 계약 예시(불합격)에만 있다
        assert "신뢰구간이 목표값을 포함하지 않는 지표(" in joined
    assert find_verdict_contradictions(texts, fs_obj.verdict) == []


def test_system_prompt_forbids_saying_a_failed_metric_does_not_affect_the_verdict():
    # 세션 C 재확인 yeast_strong: 9.2 가 비핵심 F1 미달을 '종합 판정에는 영향이 없다'고 썼다(조건부 합격의 사유인데).
    assert "미달한 지표가 있으면 그 미달이 종합 판정과 무관하다·영향이 없다고 쓰지 않는다" in prompt.SYSTEM_PROMPT
    assert "그 밖의 지표 미달은 조건부 합격의 사유다" in prompt.SYSTEM_PROMPT
