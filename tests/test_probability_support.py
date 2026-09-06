"""tests/test_probability_support.py — 확률·점수만 있는 데이터셋의 전면 지원.

ISSUES.md A-01·A-02·D-08·D-09 (2026-09-07 ★확정된 제품 결정 1).

하드 예측(y_pred/pred_labels)이 없고 확률·점수만 있는 데이터셋도 평가를 끝낼 수 있어야 한다.
- binary     : score_positive ≥ decision_threshold → 양/음성 클래스 라벨로 파생
- multiclass : prob_per_class 중 argmax → 클래스 라벨로 파생 (임계값 불필요)
- multilabel : score_per_label 각각 ≥ decision_threshold → 라벨 리스트로 파생

파생은 **모델의 실제 출력이 아니라 유도값**이므로 SPEC §0 이 요구하는 대로 파생 사실과
사용한 임계값을 응답에 실어 성적서가 인쇄할 수 있게 한다.

D-08: 역할당 1컬럼만 남기는 `{role: column}` 축약 때문에 score_per_label 다중 컬럼 중
마지막 하나만 범위 검사를 받았다. 매핑 순서만 바꿔도 같은 파일이 200/400 으로 갈렸다.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import (
    ColumnMapping,
    ColumnRole,
    DataMetadata,
    PREDICTION_ROLES_BY_TASK,
    TaskType,
    VALID_ROLES_BY_TASK,
)
from app.analysis.schemas import ConfirmMappingRequest
from app.analysis.validator import validate_mapping
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)


def _cm(column, role):
    return ColumnMapping(column=column, role=role)


def _post_evaluate(csv_text: str, request: EvaluateRequest):
    return client.post(
        "/api/evaluate",
        files={"file": ("data.csv", csv_text.encode("utf-8"), "text/csv")},
        data={"data": request.model_dump_json()},
    )


# ── 역할 표 복원 (A-02) ────────────────────────────────────────────────────

def test_probability_roles_are_valid_again():
    """multiclass 의 prob_per_class · multilabel 의 score_per_label 이 허용 역할로 돌아온다."""
    assert ColumnRole.prob_per_class in VALID_ROLES_BY_TASK[TaskType.multiclass]
    assert ColumnRole.score_per_label in VALID_ROLES_BY_TASK[TaskType.multilabel]
    assert ColumnRole.score_positive in VALID_ROLES_BY_TASK[TaskType.binary]


def test_prediction_roles_table_lists_probability_alternatives():
    """'예측 역할은 확률 역할로도 충족된다'는 규칙의 단일 출처가 존재한다."""
    for task, expected_alt in [
        (TaskType.binary, ColumnRole.score_positive),
        (TaskType.multiclass, ColumnRole.prob_per_class),
        (TaskType.multilabel, ColumnRole.score_per_label),
    ]:
        primary, alternatives = PREDICTION_ROLES_BY_TASK[task]
        assert expected_alt in alternatives, f"{task}: {expected_alt} 가 대체 역할에 없다"
        assert primary not in alternatives


# ── confirm-mapping 이 확률 전용 매핑을 통과시킨다 (A-01·A-02) ──────────────

@pytest.mark.parametrize(
    "task, truth_role, prob_mappings",
    [
        (TaskType.binary, ColumnRole.y_true, [("s", ColumnRole.score_positive)]),
        (TaskType.multiclass, ColumnRole.y_true,
         [("p_a", ColumnRole.prob_per_class), ("p_b", ColumnRole.prob_per_class)]),
        (TaskType.multilabel, ColumnRole.true_labels,
         [("s_a", ColumnRole.score_per_label), ("s_b", ColumnRole.score_per_label)]),
    ],
)
def test_confirm_mapping_accepts_probability_only(task, truth_role, prob_mappings):
    """하드 예측 없이 확률만 매핑해도 진행 가능해야 한다(종전에는 MISSING_METRIC_REQUIREMENT)."""
    mappings = [_cm("t", truth_role)] + [_cm(c, r) for c, r in prob_mappings]
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=task, column_mappings=mappings, selected_metric_ids=["M1"],
    ))
    assert resp.is_valid, [e.model_dump() for e in resp.errors]
    assert "M1" in resp.available_metric_ids


def test_confirm_mapping_still_requires_some_prediction_source():
    """정답만 있고 예측도 확률도 없으면 종전대로 막힌다."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("t", ColumnRole.y_true)],
        selected_metric_ids=["M1"],
    ))
    assert not resp.is_valid


def test_score_only_does_not_make_score_requiring_metrics_available_out_of_thin_air():
    """M9/M10/M19 는 score_positive 를 직접 요구한다 — y_pred 만 있으면 여전히 불가."""
    resp = validate_mapping(ConfirmMappingRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("t", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=["M9"],
    ))
    assert not resp.is_valid


# ── binary: threshold 파생 (A-01) ──────────────────────────────────────────

_BINARY_CSV = "y,s\n1,0.9\n1,0.2\n0,0.1\n0,0.8\n"


def _binary_request(threshold=None, metrics=("M1",)):
    return EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("y", ColumnRole.y_true), _cm("s", ColumnRole.score_positive)],
        selected_metric_ids=list(metrics),
        metadata=DataMetadata(positive_class="1", negative_class="0"),
        decision_threshold=threshold,
    )


def test_binary_score_only_derives_prediction_at_threshold():
    """score ≥ 0.5 → 양성. 4행 중 2행 적중이므로 Accuracy = 0.5 여야 한다."""
    resp = _post_evaluate(_BINARY_CSV, _binary_request(0.5))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"]["success_metrics"]["M1"] == pytest.approx(0.5)


def test_binary_threshold_actually_moves_the_result():
    """임계값이 계산에 실제로 쓰이는지 — 0.85 로 올리면 예측이 바뀌어 Accuracy 도 바뀐다.

    threshold 0.85 → 예측 [1,0,0,0] vs 정답 [1,1,0,0] → 3/4 = 0.75.
    (임계값을 무시하고 항상 0.5 를 쓰는 구현은 이 단정에서 걸린다.)
    """
    resp = _post_evaluate(_BINARY_CSV, _binary_request(0.85))
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"]["success_metrics"]["M1"] == pytest.approx(0.75)


def test_binary_derivation_defaults_to_half_when_threshold_omitted():
    """SPEC §6 의 기본값 0.5. 명시하지 않아도 파생은 성립해야 한다."""
    resp = _post_evaluate(_BINARY_CSV, _binary_request(None))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"]["success_metrics"]["M1"] == pytest.approx(0.5)
    assert body["results"]["success_metrics"]["derived_prediction"]["threshold"] == 0.5


def test_derivation_is_disclosed_in_response():
    """파생 사실·임계값·출처 컬럼이 응답에 실려야 한다(SPEC §0 — 성적서 기재 의무)."""
    body = _post_evaluate(_BINARY_CSV, _binary_request(0.5)).json()
    disclosure = body["results"]["success_metrics"]["derived_prediction"]
    assert disclosure["method"] == "threshold"
    assert disclosure["threshold"] == 0.5
    assert disclosure["source_columns"] == ["s"]
    assert disclosure["target_role"] == "y_pred"


def test_no_disclosure_when_hard_prediction_exists():
    """y_pred 가 있으면 파생하지 않는다(SPEC §1 규칙 1: 둘 다 있으면 y_pred 우선)."""
    csv = "y,p,s\n1,1,0.9\n1,0,0.2\n0,0,0.1\n0,1,0.8\n"
    req = EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[
            _cm("y", ColumnRole.y_true), _cm("p", ColumnRole.y_pred),
            _cm("s", ColumnRole.score_positive),
        ],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(positive_class="1", negative_class="0"),
        decision_threshold=0.5,
    )
    body = _post_evaluate(csv, req).json()
    assert "derived_prediction" not in body["results"]["success_metrics"]


def test_derived_labels_use_class_values_not_numbers():
    """정답이 문자열 라벨이면 파생 예측도 그 라벨이어야 한다.

    0/1 로 파생하면 sklearn 이 'mix of binary and unknown targets' 로 죽는다.
    """
    csv = "y,s\nspam,0.9\nham,0.1\nspam,0.8\nham,0.2\n"
    req = EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("y", ColumnRole.y_true), _cm("s", ColumnRole.score_positive)],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(positive_class="spam", negative_class="ham"),
        decision_threshold=0.5,
    )
    resp = _post_evaluate(csv, req)
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"]["success_metrics"]["M1"] == pytest.approx(1.0)


# ── multiclass: argmax 파생 (A-01) ─────────────────────────────────────────

_MC_CSV = "y,prob_cat,prob_dog\ncat,0.9,0.1\ndog,0.2,0.8\ncat,0.3,0.7\n"


def _mc_request(metrics=("M1",)):
    return EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("y", ColumnRole.y_true),
            _cm("prob_cat", ColumnRole.prob_per_class),
            _cm("prob_dog", ColumnRole.prob_per_class),
        ],
        selected_metric_ids=list(metrics),
        metadata=DataMetadata(detected_classes=["cat", "dog"]),
    )


def test_multiclass_prob_only_derives_by_argmax():
    """argmax → [cat, dog, dog] vs 정답 [cat, dog, cat] = 2/3."""
    resp = _post_evaluate(_MC_CSV, _mc_request())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"]["success_metrics"]["M1"] == pytest.approx(2 / 3)
    assert body["results"]["success_metrics"]["derived_prediction"]["method"] == "argmax"


def test_multiclass_argmax_refuses_when_class_names_unresolvable():
    """컬럼명에서 클래스를 확정할 수 없으면 추측하지 말고 막는다.

    컬럼 순서로 클래스를 정하는 것은 검증 불가능한 가정이다(D-07 의 선례).
    """
    csv = "y,c1,c2\ncat,0.9,0.1\ndog,0.2,0.8\n"
    req = EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("y", ColumnRole.y_true),
            _cm("c1", ColumnRole.prob_per_class),
            _cm("c2", ColumnRole.prob_per_class),
        ],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(detected_classes=["cat", "dog"]),
    )
    resp = _post_evaluate(csv, req)
    assert resp.status_code == 400, resp.text
    assert "클래스" in resp.json()["detail"]


def test_multiclass_ignores_threshold_but_says_so():
    """multiclass 는 argmax 라 임계값이 없다 — 조용히 무시하지 않고 경고를 남긴다."""
    req = _mc_request()
    req.decision_threshold = 0.7
    body = _post_evaluate(_MC_CSV, req).json()
    assert any("argmax" in w for w in body["warnings"]), body["warnings"]


# ── multilabel: 레이블별 threshold 파생 (A-01) ─────────────────────────────

_ML_CSV = (
    "t,score_a,score_b\n"
    "a|b,0.9,0.8\n"
    "a,0.7,0.2\n"
    "b,0.1,0.9\n"
)


def _ml_request(threshold=None, metrics=("M16",)):
    return EvaluateRequest(
        task_type=TaskType.multilabel,
        column_mappings=[
            _cm("t", ColumnRole.true_labels),
            _cm("score_a", ColumnRole.score_per_label),
            _cm("score_b", ColumnRole.score_per_label),
        ],
        selected_metric_ids=list(metrics),
        metadata=DataMetadata(detected_labels=["a", "b"]),
        decision_threshold=threshold,
    )


def test_multilabel_score_only_derives_per_label():
    """임계값 0.5 → [a,b], [a], [b] = 정답과 완전 일치이므로 Exact Match = 1.0."""
    resp = _post_evaluate(_ML_CSV, _ml_request(0.5))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["results"]["success_metrics"]["M16"] == pytest.approx(1.0)
    assert body["results"]["success_metrics"]["derived_prediction"]["method"] == "threshold_per_label"


def test_multilabel_accepts_per_label_thresholds():
    """레이블별로 다른 임계값을 줄 수 있다(키는 확률 컬럼명).

    score_a 임계값을 0.8 로 올리면 2행의 a(0.7)가 떨어져 예측이 [] 가 되고
    Exact Match 는 2/3 로 내려간다.
    """
    resp = _post_evaluate(_ML_CSV, _ml_request({"score_a": 0.8, "score_b": 0.5}))
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"]["success_metrics"]["M16"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("bad", [1.5, -0.1, {"score_a": 2.0}])
def test_threshold_out_of_range_is_rejected_by_the_api(bad):
    """임계값은 확률과 같은 [0,1] 구간이다. 모델을 거치지 않고 원시 JSON 으로 넣어도 422."""
    payload = {
        "task_type": "multilabel",
        "column_mappings": [
            {"column": "t", "role": "true_labels"},
            {"column": "score_a", "role": "score_per_label"},
            {"column": "score_b", "role": "score_per_label"},
        ],
        "selected_metric_ids": ["M16"],
        "metadata": {"detected_labels": ["a", "b"]},
        "decision_threshold": bad,
    }
    resp = client.post(
        "/api/evaluate",
        files={"file": ("data.csv", _ML_CSV.encode("utf-8"), "text/csv")},
        data={"data": json.dumps(payload)},
    )
    assert resp.status_code == 422, resp.text


# ── D-08: 다중 확률 컬럼 전수 범위 검사 ────────────────────────────────────

_D08_CSV = "t,p,score_bad,score_ok\na,a,1.7,0.4\nb,b,0.3,0.5\n"


@pytest.mark.parametrize("order", [["score_bad", "score_ok"], ["score_ok", "score_bad"]])
def test_all_score_columns_are_range_checked_regardless_of_mapping_order(order):
    """D-08 — 역할당 1컬럼만 남기던 축약 때문에 마지막 컬럼만 검사됐다.

    같은 파일이 매핑 순서에 따라 200/400 으로 갈렸다. 순서와 무관하게 400 이어야 한다.
    """
    req = EvaluateRequest(
        task_type=TaskType.multilabel,
        column_mappings=[
            _cm("t", ColumnRole.true_labels), _cm("p", ColumnRole.pred_labels),
        ] + [_cm(c, ColumnRole.score_per_label) for c in order],
        selected_metric_ids=["M16"],
        metadata=DataMetadata(detected_labels=["a", "b"]),
    )
    resp = _post_evaluate(_D08_CSV, req)
    assert resp.status_code == 400, resp.text
    assert "score_bad" in resp.json()["detail"]


def test_multiclass_prob_columns_all_range_checked():
    """multiclass prob_per_class 도 같은 규칙(종전에도 리스트로 모았지만 회귀 방지)."""
    csv = "y,prob_cat,prob_dog\ncat,1.4,0.1\ndog,0.2,0.8\n"
    req = EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[
            _cm("y", ColumnRole.y_true),
            _cm("prob_cat", ColumnRole.prob_per_class),
            _cm("prob_dog", ColumnRole.prob_per_class),
        ],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(detected_classes=["cat", "dog"]),
    )
    resp = _post_evaluate(csv, req)
    assert resp.status_code == 400, resp.text
    assert "prob_cat" in resp.json()["detail"]


# ── D-11: 오류 메시지의 행 번호 ────────────────────────────────────────────

def test_score_range_error_reports_one_based_row_number():
    """D-11 — 0-based 인덱스를 '{n}번째 행'으로 인쇄하던 문제.

    첫 데이터 행(파일의 2번째 줄)이 위반이면 '1번째 행'이라고 말해야 한다.
    """
    csv = "y,s\n1,1.7\n0,0.3\n"
    req = EvaluateRequest(
        task_type=TaskType.binary,
        column_mappings=[_cm("y", ColumnRole.y_true), _cm("s", ColumnRole.score_positive)],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(positive_class="1", negative_class="0"),
    )
    resp = _post_evaluate(csv, req)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "1번째 행" in detail and "0번째 행" not in detail
