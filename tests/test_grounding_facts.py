"""tests/test_grounding_facts.py — 숫자 허용 목록 확장(S4b, 점검 생-4·재-1).

흔한 표현이 숫자 검문에 걸려 7·8·9절 전체가 규칙 문장이 되던 것(S1 실호출의 "합계 57건", "−0.0213", "β=2",
"class_3", "95번째 백분위")을 풀고, 판정 기준값은 그 판정이 켜졌을 때만 허용한다. 판정과 반대로 읽히는 짧은
반올림 표기는 뺀다.
"""
from app.narrative.grounding import build_number_whitelist, verify_grounding
from app.narrative.schemas import (
    ConfusionCountFact,
    ConfusionFact,
    FactSheet,
    LatencyFact,
    MetricFact,
    NarrativeFactsFact,
    PerClassFact,
)


def _fs(**over) -> FactSheet:
    base = dict(verdict="PASS", score=100.0, n_samples=200)
    base.update(over)
    return FactSheet(**base)


def _passes(fs: FactSheet, text: str) -> bool:
    return verify_grounding([text], build_number_whitelist(fs, [], {}, edition=3), edition=3).passed


def _metric(**over) -> MetricFact:
    base = dict(metric_id="M1", display_name="Accuracy", value=0.8213, threshold=0.8, status="pass", margin=0.0213)
    base.update(over)
    return MetricFact(**base)


def test_unicode_minus_and_magnitude_of_signed_difference():
    fs = _fs(metrics=[_metric(value=0.7787, margin=-0.0213, status="fail")])
    assert _passes(fs, "목표와의 차이는 −0.0213이다.")
    assert _passes(fs, "목표보다 0.0213 낮다.")
    assert not _passes(fs, "목표와의 차이는 −0.0214이다.")


def test_beta_is_allowed_only_when_sent():
    # β 는 다른 사실에서 우연히 나오지 않는 값으로 본다(마진 0.0213 의 백분율 반올림이 '2' 다).
    with_beta = _fs(metrics=[_metric(metric_id="M5", display_name="F-beta Score", beta=7.0)])
    assert _passes(with_beta, "F-beta(β=7) 점수는 0.8213이다.")
    assert not _passes(_fs(metrics=[_metric(metric_id="M5", display_name="F-beta Score")]), "β=7 기준이다.")


def test_numbers_inside_class_names():
    fs = _fs(per_class=[PerClassFact(label="class_3", f1=0.5, support=10), PerClassFact(label="v2 결함", f1=0.6, support=12)])
    assert _passes(fs, "class_3 과 v2 결함 클래스의 F1 은 0.5, 0.6 이다.")


def test_common_sums_per_class_and_matrix_margins():
    counts = [ConfusionCountFact(label="a", tp=10, fp=27, fn=30, tn=133)]
    fs = _fs(confusion_counts=counts)
    assert _passes(fs, "a 클래스의 오류는 합계 57건이다.")  # fp + fn
    assert _passes(fs, "a 로 예측한 것은 37건, 실제 a 는 40건이다.")
    matrix = _fs(confusion=ConfusionFact(labels=["x", "y"], matrix=[[50, 7], [3, 40]]))
    assert _passes(matrix, "실제 x 는 57건, y 로 예측한 것은 47건이다.")
    assert not _passes(matrix, "합계 58건이다.")


def test_latency_percentile_names_only_with_latency():
    lat = LatencyFact(available=True, mean=12.0, p50=10.0, p95=30.0, p99=45.0)
    assert _passes(_fs(latency=lat), "95번째 백분위 지연은 30ms 이다.")
    assert not _passes(_fs(), "95번째 백분위 지연이다.")


def _facts(**over) -> NarrativeFactsFact:
    return NarrativeFactsFact(**over)


def test_facts_numbers_are_allowed_with_magnitudes():
    facts = _facts(
        declared_risk_errors={"side": "fn", "count": 43, "total_errors": 99, "share": 0.434343434, "share_text": "0.4343"},
        prediction_bias={"unit": "positive_class", "rows": [{"label": "1", "predicted": 95, "actual": 108, "difference": -13}]},
        top_confusion_pairs=[{"actual": "dog", "predicted": "cat", "count": 28, "share_of_errors": 0.210526316}],
        n_classes=3,
    )
    fs = _fs(facts=facts)
    assert _passes(fs, "FN 은 전체 오류 99건 중 43건(43.4%)이다.")
    assert _passes(fs, "양성을 실제보다 13건 적게 예측했다(−13).")
    assert _passes(fs, "dog 을 cat 으로 잘못 본 것이 28건으로 전체 오분류의 21.1%다.")
    assert _passes(fs, "3개 클래스")
    assert not _passes(fs, "FN 은 44건이다.")


def test_thresholds_only_when_their_judgement_is_on():
    weak = {
        "label": "c", "f1": 0.3, "support": 12, "share": 0.06, "share_text": "0.06",
        "macro_f1": 0.5, "gap_from_macro": 0.2, "gap_text": "0.2", "low_support": True, "far_below_macro": True,
    }
    on = _fs(facts=_facts(weakest_class=weak))
    assert _passes(on, "표본이 30건 미만이고 매크로 F1 보다 0.1 이상 낮다.")
    off = _fs(facts=_facts())
    assert not _passes(off, "표본이 30건 미만이다.")
    shift = _fs(facts=_facts(distribution_shift=[{
        "label": "a", "training_ratio": 0.7, "evaluation_ratio": 0.5, "difference": 0.2,
        "difference_text": "0.2", "flagged": True,
    }]))
    assert _passes(shift, "학습과 평가의 비율 차가 10%p 이상이다.")
    assert not _passes(off, "비율 차가 10%p 이상이다.")


def test_short_rounding_that_reads_against_the_judgement_is_dropped():
    # 미달 0.79996 은 셋째 자리로 0.8 — '0.8' 은 값으로 허용하지 않는다(표기는 0.79996). 목표 0.9 는 반올림해도
    # 0.8 이 나오지 않는 값으로 골랐다(목표 0.85 는 한 자리 반올림이 0.8 이라 다른 길로 허용된다).
    fs = _fs(metrics=[_metric(value=0.79996, threshold=0.9, status="fail", value_text="0.79996", margin=-0.10004)])
    assert _passes(fs, "Accuracy 는 0.79996 이다.")
    assert not _passes(fs, "Accuracy 는 0.8 이다.")
    # 평소 자릿수 표기면 종전처럼 반올림 표기도 허용한다.
    plain = _fs(metrics=[_metric(value=0.8213, value_text="0.821")])
    assert _passes(plain, "Accuracy 는 0.82 이다.")


def test_display_texts_of_engine_values_are_allowed():
    from app.narrative.schemas import F1ThresholdGainFact, ImbalanceEffectFact

    gain = F1ThresholdGainFact(max_f1=0.6, current_f1=0.5000400, gain=0.09996, best_threshold=0.3, notable=False, gain_text="0.09996")
    ie = ImbalanceEffectFact(
        level="mild", imbalance_ratio=1.5, accuracy_minus_macro_f1=0.0001, majority_minus_minority_f1=-0.0149,
        majority_class="0", minority_class="1",
        display={"imbalance_ratio": "1.49996", "accuracy_minus_macro_f1": "0.0001", "majority_minus_minority_f1": "-0.0149"},
    )
    fs = _fs(f1_threshold_gain=gain, imbalance_effect=ie)
    assert _passes(fs, "향상 여지는 0.09996 이고 불균형 비율은 1.49996, 소수 클래스 F1 이 0.0149 높다.")


def test_imbalance_effect_four_values_are_allowed_only_when_sent():
    """엔진이 새로 주는 네 값(정확도·매크로 F1·다수/소수 F1, 사용자 결정 2026-10-22)은 사실 시트에 실리면 허용 — 옛 응답(키 없음)은 아님."""
    from app.narrative.schemas import ImbalanceEffectFact

    base = dict(
        level="affected", imbalance_ratio=3.5, accuracy_minus_macro_f1=0.0056, majority_minus_minority_f1=0.1,
        majority_class="neg", minority_class="pos",
    )
    text = "정확도 0.5677, 매크로 F1 0.5621, 다수 F1 0.6123, 소수 F1 0.5123 이다."
    old = ImbalanceEffectFact(**base, display={
        "imbalance_ratio": "3.5", "accuracy_minus_macro_f1": "0.0056", "majority_minus_minority_f1": "0.1"})
    assert not _passes(_fs(imbalance_effect=old), text)
    new = ImbalanceEffectFact(
        **base, accuracy=0.5677, macro_f1=0.5621, majority_f1=0.6123, minority_f1=0.5123,
        display={"imbalance_ratio": "3.5", "accuracy_minus_macro_f1": "0.0056", "majority_minus_minority_f1": "0.1",
                 "accuracy": "0.5677", "macro_f1": "0.5621", "majority_f1": "0.6123", "minority_f1": "0.5123"},
    )
    assert _passes(_fs(imbalance_effect=new), text)
    # display 없이 숫자 칸만 있어도(표시 문자열이 없는 경우) 값으로 허용한다.
    bare = ImbalanceEffectFact(**base, accuracy=0.5677, macro_f1=0.5621, majority_f1=0.6123, minority_f1=0.5123)
    assert _passes(_fs(imbalance_effect=bare), text)


def test_imbalance_effect_four_values_use_their_display_strings():
    """네 값의 표시 문자열이 원값의 평소 표기와 다를 때(판정 쪽으로 자릿수를 늘린 꼴) 그 표시 문자열이 허용 목록에 든다 —
    값만 보고 넣으면 이 표기가 빠진다(세션 B 변이 M57·M58). 값마다 따로 본다."""
    from app.narrative.schemas import ImbalanceEffectFact

    shown = {"accuracy": "0.56771", "macro_f1": "0.56214", "majority_f1": "0.61237", "minority_f1": "0.51238"}
    ie = ImbalanceEffectFact(
        level="affected", imbalance_ratio=3.5, accuracy_minus_macro_f1=0.0056, majority_minus_minority_f1=0.1,
        majority_class="neg", minority_class="pos",
        accuracy=0.567712, macro_f1=0.562143, majority_f1=0.612374, minority_f1=0.512381,
        display={"imbalance_ratio": "3.5", "accuracy_minus_macro_f1": "0.0056", "majority_minus_minority_f1": "0.1", **shown},
    )
    fs = _fs(imbalance_effect=ie)
    for key, text in shown.items():
        assert _passes(fs, f"값은 {text}이다."), key
        # 표시 문자열이 길면 그보다 짧은 반올림 표기는 넣지 않는다(_add_judged) — 값만으로 넣으면 이 표기가 들어온다.
        assert not _passes(fs, f"값은 {round(float(text), 3):.3f}이다."), key
