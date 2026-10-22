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
