"""정답·예측 라벨의 자료형 맞추기 — 라운드 C 검토에서 재현한 회귀를 고정한다.

1) 불리언 예측 + 정수 정답(또는 그 반대). `proba > 0.5` 결과를 그대로 저장하면 예측이 True/False 다. 종전(b13e55c)
   전처리는 예측을 정답 자료형으로 캐스팅해 M1 0.75 를 냈는데, 표기 맞추기(S2)가 그 캐스팅을 지우면서
   normalize_label(True)='True' 와 '1' 이 다른 클래스가 되어 이진은 400('클래스가 4개'), 다중은 Accuracy 0.0 이
   성적서에 나갔다. 아래 기대값은 같은 데이터를 b13e55c 의 engine.evaluate 에 넣어 얻은 값이다.
2) 정답 클래스가 하나뿐인 이진 데이터 — 검증의 'Binary class system error' 가 정답∪예측을 세면서 경고가 통과로
   바뀌었다. 정답 클래스 수 < 2 이면 종전처럼 경고(found 1), 정답∪예측 > 2 이면 오류.
3) 숫자 라벨이 문자열로 바뀌지 않는다 — 정수 라벨을 object 문자열로 바꾸면 sklearn 이 지표마다 문자열 배열을
   정렬해 10MB CSV 평가가 약 20배 느려졌다(시간 회귀는 test_label_speed.py).
"""
import io

import numpy as np
import pandas as pd
import pytest

from app.analysis.validation_checks import check_binary, check_class_mismatch
from app.evaluation.engine import evaluate
from app.evaluation import preprocessor
from app.evaluation.preprocessor import preprocess_data

ROWS = [(1, "True", 0.9), (0, "False", 0.2), (1, "False", 0.4), (0, "False", 0.1)] * 25


def _int_truth_bool_pred() -> pd.DataFrame:
    csv = "label,pred,prob\n" + "".join(f"{t},{p},{s}\n" for t, p, s in ROWS)
    df = pd.read_csv(io.StringIO(csv))
    assert df["label"].dtype == np.int64 and df["pred"].dtype == bool
    return df


def _bool_truth_int_pred() -> pd.DataFrame:
    return _int_truth_bool_pred().rename(columns={"label": "pred", "pred": "label"})


def _m(score=False):
    m = [{"column": "label", "role": "y_true"}, {"column": "pred", "role": "y_pred"}]
    return m + ([{"column": "prob", "role": "score_positive"}] if score else [])


BIN = ["M1", "M2", "M3", "M4", "M7", "M21", "M22"]


def test_int_truth_bool_pred_binary_matches_old_engine():
    r = evaluate(_int_truth_bool_pred(), _m(score=True), "binary", BIN)
    assert "error" not in r, r.get("error")
    assert (r["M1"], r["M2"], r["M3"], r["M7"]) == (0.75, 1.0, 0.5, 1.0)
    assert r["M4"] == pytest.approx(2 / 3)
    assert r["M21"] == {"type": "multiclass_or_binary", "labels": ["0", "1"], "matrix": [[50, 0], [25, 25]]}
    assert set(r["M22"]) == {"0", "1", "accuracy", "macro avg", "weighted avg"}
    assert r["_metadata"]["class_distribution"] == {"0": 50, "1": 50}


def test_int_truth_bool_pred_binary_with_positive_class_from_metadata():
    r = evaluate(_int_truth_bool_pred(), _m(), "binary", BIN, positive_class="1")
    assert (r["M1"], r["M2"], r["M3"]) == (0.75, 1.0, 0.5)
    assert r["confusion_counts"]["positive_class"] == "1"


def test_bool_truth_int_pred_keeps_truth_notation():
    # 종전처럼 예측이 정답 표기(True/False)를 따른다 — metadata 의 양성 'True' 가 그대로 데이터에서 찾아진다.
    for positive in (None, "True"):
        r = evaluate(_bool_truth_int_pred(), _m(), "binary", BIN, positive_class=positive)
        assert "error" not in r, r.get("error")
        assert (r["M1"], r["M2"], r["M3"]) == (0.75, 0.5, 1.0)
        assert r["M7"] == pytest.approx(2 / 3)
        assert r["M21"]["labels"] == ["False", "True"]
        assert r["M21"]["matrix"] == [[50, 25], [0, 25]]
        assert r["confusion_counts"]["positive_class"] == "True"
        assert r["_metadata"]["class_distribution"] == {"False": 75, "True": 25}


@pytest.mark.parametrize("make, labels", [(_int_truth_bool_pred, ["0", "1"]), (_bool_truth_int_pred, ["False", "True"])])
def test_mixed_bool_multiclass_is_not_zero(make, labels):
    r = evaluate(make(), _m(), "multiclass", ["M1", "M4", "M21"])
    assert r["M1"] == 0.75
    assert r["M4"] == pytest.approx(0.7333333333333334)
    assert r["M21"]["labels"] == labels


def test_both_bool_keep_true_false_names():
    df = pd.DataFrame({"label": [True, False, True, False], "pred": [True, False, False, False]})
    r = evaluate(df, _m(), "binary", ["M1", "M21"])
    assert r["M1"] == 0.75
    assert r["M21"]["labels"] == ["False", "True"]
    assert r["_metadata"]["class_distribution"] == {"False": 2, "True": 2}


def test_bool_with_string_zero_one_column():
    df = pd.DataFrame({"label": ["1", "0", "1", "0"], "pred": [True, False, False, False]})
    r = evaluate(df, _m(), "binary", ["M1", "M21"])
    assert r["M1"] == 0.75
    assert r["M21"]["labels"] == ["0", "1"]


def test_bool_with_other_strings_is_left_alone():
    # 'cat'/'dog' 정답과 불리언 예측은 맞출 근거가 없다 — 종전처럼 클래스가 넷이라 이진은 막힌다.
    df = pd.DataFrame({"label": ["cat", "dog"], "pred": [True, False]})
    out = preprocessor.align_boolean_labels(df, {"y_true": "label", "y_pred": "pred"})
    assert out["pred"].tolist() == [True, False]


def test_validation_agrees_with_evaluation_on_mixed_bool():
    df = _int_truth_bool_pred()
    mapping = {"y_true": "label", "y_pred": "pred"}
    df = preprocessor.align_boolean_labels(df, mapping)
    binary = {i.name: i for i in check_binary(df, mapping, {"score_positive": ["prob"]})}
    assert binary["Binary class system error"].status == "pass"
    mismatch = check_class_mismatch(df, mapping)
    assert mismatch[0].status == "pass", mismatch[0].result


def test_validate_endpoint_does_not_block_mixed_bool():
    from fastapi.testclient import TestClient
    import json

    from app.main import app

    csv = "label,pred\n" + "".join(f"{t},{p}\n" for t, p, _ in ROWS)
    data = json.dumps({
        "task_type": "binary",
        "column_mappings": [{"column": "label", "role": "y_true", "sample_values": []},
                            {"column": "pred", "role": "y_pred", "sample_values": []}],
        "selected_metric_ids": ["M1"],
        "metadata": {
            "positive_class": "1", "negative_class": "0", "positive_class_ambiguous": False,
            "detected_classes": [], "detected_labels": [], "class_distribution": {}, "column_unique_values": {},
        },
        "beta": 1.0,
    })
    body = TestClient(app).post(
        "/api/validate-data", files={"file": ("b.csv", csv.encode(), "text/csv")}, data={"data": data},
    ).json()
    bad = [(d["name"], d["result"]) for d in body["validation_details"] if d["status"] in ("warning", "error")]
    assert not bad, bad
    assert body["class_distribution"] == {"0": 50, "1": 50}


# ── 이진 클래스 수 검사: 정답 클래스 하나 → 경고, 정답∪예측 셋 이상 → 오류 ──

def _binary_class_item(truth, pred):
    df = pd.DataFrame({"t": truth, "p": pred})
    items = check_binary(df, {"y_true": "t", "y_pred": "p"}, {})
    return next(i for i in items if i.name == "Binary class system error")


def test_single_true_class_is_still_a_warning():
    item = _binary_class_item([1] * 50, [0, 1] * 25)
    assert (item.status, item.result) == ("warning", "Expected 2 classes, found 1")
    assert item.handling == "Reported only — evaluation continues"


def test_extra_predicted_class_is_an_error():
    item = _binary_class_item([0, 1, 0, 1], [0, 1, 2, 1])
    assert (item.status, item.result) == ("error", "Expected 2 classes, found 3")


def test_single_true_class_with_two_extra_predictions_is_an_error():
    item = _binary_class_item([1, 1, 1], [0, 1, 2])
    assert item.status == "error"


def test_two_true_classes_pass():
    item = _binary_class_item([0, 1, 0, 1], [1, 1, 0, 0])
    assert item.status == "pass"


# ── 숫자 라벨은 숫자로 남는다(성능) ──

def test_integer_labels_stay_numeric_after_preprocess():
    df = pd.DataFrame({"t": [0, 1, 1, 0], "p": [0.0, 1.0, 0.0, 0.0]})
    out, _ = preprocess_data(df, [{"column": "t", "role": "y_true"}, {"column": "p", "role": "y_pred"}],
                             "binary", selected_metric_ids=["M1"])
    assert out["t"].dtype == np.int64 and out["p"].dtype == np.int64


def test_string_labels_are_still_normalized():
    df = pd.DataFrame({"t": [" cat", "dog", "1.0"], "p": ["cat", "dog", "1"]})
    out, _ = preprocess_data(df, [{"column": "t", "role": "y_true"}, {"column": "p", "role": "y_pred"}],
                             "multiclass", selected_metric_ids=["M1"])
    assert out["t"].tolist() == ["cat", "dog", "1"] and out["p"].tolist() == ["cat", "dog", "1"]


def test_derived_prediction_with_class_only_in_probability_names():
    # 정수 정답 0·1·2 와, metadata 의 클래스에만 있는 prob_3. 파생 예측에 이름 '3'(문자열)이 들어가도 정답과 한 표기로
    # 맞춰져야 한다 — 고치기 전(문자열 경로)과 같은 값이다.
    df = pd.DataFrame({
        "t": [0, 1, 2, 0, 1, 2],
        "prob_0": [.7, .1, .1, .1, .1, .1], "prob_1": [.1, .7, .1, .1, .1, .1],
        "prob_2": [.1, .1, .7, .1, .1, .1], "prob_3": [.1, .1, .1, .7, .7, .7],
    })
    m = [{"column": "t", "role": "y_true"}] + [{"column": f"prob_{k}", "role": "prob_per_class"} for k in range(4)]
    r = evaluate(df, m, "multiclass", ["M1", "M4", "M21"], metadata={"detected_classes": ["0", "1", "2", "3"]})
    assert r["M1"] == 0.5
    assert r["M4"] == pytest.approx(2 / 3)
    assert r["M21"]["labels"] == ["0", "1", "2", "3"]
    assert r["M21"]["matrix"] == [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]]
