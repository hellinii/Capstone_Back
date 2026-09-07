"""경계 조건에서 지표·검증이 조용히 틀린 값을 내는지 검증하는 기대값 테스트.

기존 test_evaluator.py 는 "float 인가 / 0~1 인가"만 단정하고 픽스처가 무작위 노이즈라,
공식이 틀려도 값이 그럴듯해서 통과한다. 여기서는 정답을 손으로 계산할 수 있는
작은 데이터로 **기대값 자체**를 고정한다.

ISSUES.md 대응: C-01, C-02, A-07, D-05, D-07, D-10
"""
import pandas as pd
import pytest

from app.analysis import validation_checks
from app.evaluation.metrics import binary, common, multilabel
from app.evaluation.preprocessor import preprocess_data


def _binary_mapping(positive_class=1):
    return {"y_true": "t", "y_pred": "p", "_task_type": "binary",
            "_positive_class": positive_class, "_beta": 1.0}


# ── C-02: 혼동행렬이 2x2 가 아닐 때 0.0 으로 뭉개지 않는다 ────────────────────

def test_specificity_all_negative_and_all_correct():
    """정답이 전부 음성이고 전부 맞혔으면 Specificity 는 1.0, FPR 은 0.0 이다.

    (희귀 양성 필터링 데이터에서 흔한 형태. 종전에는 혼동행렬이 1x1 이라 둘 다 0.0)
    """
    df = pd.DataFrame({"t": [0, 0, 0, 0], "p": [0, 0, 0, 0]})
    assert binary.calculate_specificity(df, _binary_mapping()) == 1.0
    assert binary.calculate_fpr(df, _binary_mapping()) == 0.0


def test_specificity_all_negative_with_false_positives():
    """음성만 있는 데이터에서 2건을 양성으로 오탐하면 Specificity 0.5 / FPR 0.5."""
    df = pd.DataFrame({"t": [0, 0, 0, 0], "p": [0, 1, 0, 1]})
    assert binary.calculate_specificity(df, _binary_mapping()) == 0.5
    assert binary.calculate_fpr(df, _binary_mapping()) == 0.5


def test_specificity_unknown_label_in_pred_is_not_positive():
    """y_pred 에 미지 라벨이 섞여도 0.0 으로 뭉개지지 않는다.

    t=[1,0,1,0], p=[1,0,2,0] → 양성=1 기준 음성 2건 모두 음성 예측 → Specificity 1.0
    """
    df = pd.DataFrame({"t": [1, 0, 1, 0], "p": [1, 0, 2, 0]})
    assert binary.calculate_specificity(df, _binary_mapping()) == 1.0
    assert binary.calculate_fpr(df, _binary_mapping()) == 0.0


# ── A-07: M7/M8 이 positive_class 를 반영한다 ────────────────────────────────

def test_specificity_respects_positive_class():
    """양성 클래스를 뒤집으면 Specificity/FPR 도 뒤집힌 기준으로 계산된다.

    t=[1,1,1,0,0], p=[1,1,0,0,1]
      양성=1 → 음성은 t==0 인 2건, 그중 1건을 양성으로 오탐 → Spec 0.5, FPR 0.5
      양성=0 → 음성은 t==1 인 3건, 그중 2건을 양성(=1)로 예측 → Spec 2/3, FPR 1/3
    """
    df = pd.DataFrame({"t": [1, 1, 1, 0, 0], "p": [1, 1, 0, 0, 1]})

    assert binary.calculate_specificity(df, _binary_mapping(1)) == 0.5
    assert binary.calculate_fpr(df, _binary_mapping(1)) == 0.5

    assert binary.calculate_specificity(df, _binary_mapping(0)) == pytest.approx(2 / 3)
    assert binary.calculate_fpr(df, _binary_mapping(0)) == pytest.approx(1 / 3)


# ── C-01: 빈 레이블 샘플이 M17 을 깎지 않는다 ────────────────────────────────

def test_multilabel_perfect_prediction_scores_perfect():
    """정답과 예측이 완전히 같으면 빈 레이블 행이 있어도 모든 지표가 만점이어야 한다."""
    df = pd.DataFrame({"t": ["A|B", "A", "", "", "B"], "p": ["A|B", "A", "", "", "B"]})
    m = {"true_labels": "t", "pred_labels": "p", "_task_type": "multilabel", "_beta": 1.0}

    assert multilabel.calculate_hamming_loss(df, m) == 0.0
    assert multilabel.calculate_exact_match_ratio(df, m) == 1.0
    assert multilabel.calculate_jaccard_index(df, m) == 1.0


def test_multilabel_jaccard_still_penalizes_real_mismatch():
    """빈 레이블 처리를 완화해도 진짜 불일치는 그대로 감점된다.

    행별 Jaccard: A|B vs A → 1/2, A vs A → 1, '' vs '' → 1(둘 다 없음), B vs C → 0
    평균 = (0.5 + 1 + 1 + 0) / 4 = 0.625
    """
    df = pd.DataFrame({"t": ["A|B", "A", "", "B"], "p": ["A", "A", "", "C"]})
    m = {"true_labels": "t", "pred_labels": "p", "_task_type": "multilabel", "_beta": 1.0}
    assert multilabel.calculate_jaccard_index(df, m) == pytest.approx(0.625)


# ── D-05: 확률을 y_pred 로 매핑하면 조용히 절단되지 않고 차단된다 ─────────────

def test_probability_mapped_as_pred_is_rejected():
    """y_pred 에 확률이 매핑되면 int 절단으로 넘어가지 않고 에러로 막는다."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "t": [1, 0, 1, 0], "p": [0.6, 0.4, 0.9, 0.2]})
    mappings = [{"column": "id", "role": "sample_id"},
                {"column": "t", "role": "y_true"},
                {"column": "p", "role": "y_pred"}]

    with pytest.raises(ValueError, match="확률"):
        preprocess_data(df, mappings, "binary")


def test_integer_valued_float_pred_is_still_accepted():
    """소수점 표기지만 값이 정수인 예측(1.0/0.0)은 정상 캐스팅된다(오탐 방지)."""
    df = pd.DataFrame({"id": [1, 2, 3, 4], "t": [1, 0, 1, 0], "p": [1.0, 0.0, 1.0, 0.0]})
    mappings = [{"column": "id", "role": "sample_id"},
                {"column": "t", "role": "y_true"},
                {"column": "p", "role": "y_pred"}]

    out, _ = preprocess_data(df, mappings, "binary")
    assert out["p"].tolist() == [1, 0, 1, 0]
    assert common.calculate_accuracy(out, _binary_mapping()) == 1.0


# ── D-10: 단일 라벨은 형식 오류가 아니다 ────────────────────────────────────

def _multilabel_check(true_values):
    df = pd.DataFrame({"t": true_values, "p": true_values})
    items = validation_checks.check_multilabel(df, {"true_labels": "t", "pred_labels": "p"}, {})
    return next(i for i in items if i.name == "Label format mismatch")


def test_single_label_rows_are_not_format_errors():
    """라벨이 하나뿐인 행은 구분자가 없는 게 정상이므로 형식 오류가 아니다."""
    assert _multilabel_check(["A|B", "A", "B", "A|C"]).status == "pass"


def test_multilabel_format_check_still_passes_on_delimited_rows():
    assert _multilabel_check(["A|B", "A|C", "B|C"]).status == "pass"


# ── D-07: argmax 검사가 컬럼 작명에 좌우되지 않는다 ─────────────────────────

def _argmax_item(df, prob_cols):
    items = validation_checks.check_multiclass(df, {"y_true": "t", "y_pred": "p"},
                                              {"prob_per_class": prob_cols})
    return next((i for i in items if i.name == "Argmax and y_pred mismatch"), None)


def test_argmax_check_does_not_false_alarm_on_other_naming():
    """확률 컬럼 이름이 'prob_<클래스>' 규칙을 따르지 않으면 허위 경고를 내지 않는다."""
    df = pd.DataFrame({"t": ["cat", "dog", "cat"], "p": ["cat", "dog", "cat"],
                       "p_cat": [0.9, 0.1, 0.8], "p_dog": [0.1, 0.9, 0.2]})
    item = _argmax_item(df, ["p_cat", "p_dog"])
    assert item is None or item.status != "warning", (
        "클래스명을 컬럼명에서 알아낼 수 없으면 검사를 건너뛰어야 한다"
    )


def test_argmax_check_still_detects_real_mismatch_when_resolvable():
    """컬럼명으로 클래스를 알아낼 수 있으면 진짜 불일치는 계속 잡는다."""
    df = pd.DataFrame({"t": ["cat", "dog", "cat"], "p": ["cat", "dog", "dog"],
                       "prob_cat": [0.9, 0.1, 0.8], "prob_dog": [0.1, 0.9, 0.2]})
    item = _argmax_item(df, ["prob_cat", "prob_dog"])
    assert item is not None and item.status == "warning" and item.result == "1 rows"
