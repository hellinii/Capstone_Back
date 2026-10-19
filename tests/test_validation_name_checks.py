"""검증 단계의 새 항목 — 확률 컬럼 이름(M3)·지연시간 단위(M4).

M3: 다중·멀티에서 y_pred 없이 확률만 매핑하면 평가가 컬럼 이름으로 클래스를 확정한다. 이름이 규칙
('접두어 + 클래스 이름')을 어기면 종전에는 검증을 통과한 뒤 평가에서 400 으로 막혔다. 이제 검증이
같은 함수로 미리 보고 error 항목을 낸다.
M4: 지연시간 역할은 ms 로 읽는다. 값이 모두 1 미만이면 초로 적은 것으로 보여 warning 을 낸다.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analysis import validation_checks as checks
from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.preprocessor import PROB_COLUMN_PREFIXES, probability_names_problem
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)

NAME = checks.PROBABILITY_COLUMN_NAMES_CHECK
UNIT = checks.LATENCY_UNIT_CHECK


def _validate(csv: str, task: str, mappings: list[tuple[str, str]], metrics: list[str]):
    req = EvaluateRequest(
        task_type=TaskType(task),
        column_mappings=[ColumnMapping(column=c, role=ColumnRole(r)) for c, r in mappings],
        selected_metric_ids=metrics,
        metadata=DataMetadata(),
    )
    r = client.post(
        "/api/validate-data",
        files={"file": ("d.csv", csv.encode(), "text/csv")},
        data={"data": req.model_dump_json()},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _items(body, name):
    return [d for d in body["validation_details"] if d["name"] == name]


def test_contract_names_and_prefixes():
    """프론트가 대조하는 값 — 바꾸면 프론트 이름 표·안내도 같이 바꾼다."""
    assert NAME == "Probability column names"
    assert UNIT == "Latency unit"
    assert PROB_COLUMN_PREFIXES == ("prob_", "probability_", "score_", "p_")


# ── M3 ────────────────────────────────────────────────────────────────────────

_MC_BAD = "id,y_true,prob_a,prob_b\n1,cat,0.9,0.1\n2,dog,0.2,0.8\n3,cat,0.7,0.3\n"
_MC_OK = "id,y_true,prob_cat,prob_dog\n1,cat,0.9,0.1\n2,dog,0.2,0.8\n3,cat,0.7,0.3\n"


def test_multiclass_unresolvable_probability_names_is_one_error():
    body = _validate(_MC_BAD, "multiclass",
                     [("id", "sample_id"), ("y_true", "y_true"), ("prob_a", "prob_per_class"), ("prob_b", "prob_per_class")],
                     ["M1"])
    items = _items(body, NAME)
    assert len(items) == 1
    assert items[0]["status"] == "error" and items[0]["group"] == "multiclass"
    assert "prob_a" in items[0]["result"]
    assert body["error_count"] >= 1


def test_multiclass_matching_names_has_no_item():
    body = _validate(_MC_OK, "multiclass",
                     [("y_true", "y_true"), ("prob_cat", "prob_per_class"), ("prob_dog", "prob_per_class")],
                     ["M1"])
    assert _items(body, NAME) == []


def test_name_check_skipped_when_y_pred_is_mapped():
    csv = "y_true,y_pred,prob_a,prob_b\ncat,cat,0.9,0.1\ndog,dog,0.2,0.8\n"
    body = _validate(csv, "multiclass",
                     [("y_true", "y_true"), ("y_pred", "y_pred"), ("prob_a", "prob_per_class"), ("prob_b", "prob_per_class")],
                     ["M1"])
    assert _items(body, NAME) == []


def test_name_check_skipped_when_metrics_need_no_prediction():
    """M23(불균형비)만 고르면 예측을 만들지 않으므로 이름을 보지 않는다(평가도 400 을 내지 않는다)."""
    body = _validate(_MC_BAD, "multiclass",
                     [("y_true", "y_true"), ("prob_a", "prob_per_class"), ("prob_b", "prob_per_class")],
                     ["M23"])
    assert _items(body, NAME) == []


def test_multilabel_unresolvable_names_is_error_and_matching_is_clean():
    bad = "true_labels,score_x,score_y\nsports|news,0.9,0.1\nnews,0.2,0.8\n"
    ok = "true_labels,score_sports,score_news\nsports|news,0.9,0.1\nnews,0.2,0.8\n"
    maps_bad = [("true_labels", "true_labels"), ("score_x", "score_per_label"), ("score_y", "score_per_label")]
    maps_ok = [("true_labels", "true_labels"), ("score_sports", "score_per_label"), ("score_news", "score_per_label")]

    items = _items(_validate(bad, "multilabel", maps_bad, ["M16"]), NAME)
    assert len(items) == 1 and items[0]["status"] == "error" and items[0]["group"] == "multilabel"
    assert _items(_validate(ok, "multilabel", maps_ok, ["M16"]), NAME) == []


def test_binary_never_checks_names():
    csv = "y_true,whatever\n1,0.9\n0,0.2\n"
    body = _validate(csv, "binary", [("y_true", "y_true"), ("whatever", "score_positive")], ["M1"])
    assert _items(body, NAME) == []


def test_validation_and_evaluation_agree_on_names():
    """검증이 막는 이름은 평가도 400 으로 막고, 검증이 통과시킨 이름은 평가도 통과한다(같은 판정)."""
    import json

    def _evaluate(csv, mappings):
        req = EvaluateRequest(
            task_type=TaskType.multiclass,
            column_mappings=[ColumnMapping(column=c, role=ColumnRole(r)) for c, r in mappings],
            selected_metric_ids=["M1"],
            metadata=DataMetadata(),
        )
        return client.post("/api/evaluate", files={"file": ("d.csv", csv.encode(), "text/csv")},
                           data={"data": req.model_dump_json()})

    bad_maps = [("y_true", "y_true"), ("prob_a", "prob_per_class"), ("prob_b", "prob_per_class")]
    ok_maps = [("y_true", "y_true"), ("prob_cat", "prob_per_class"), ("prob_dog", "prob_per_class")]
    assert _evaluate(_MC_BAD, bad_maps).status_code == 400
    assert _evaluate(_MC_OK, ok_maps).status_code == 200


def test_probability_names_problem_pure():
    df = pd.DataFrame({"t": ["cat", "dog"], "prob_a": [0.9, 0.2], "prob_b": [0.1, 0.8]})
    roles = {"y_true": ["t"], "prob_per_class": ["prob_a", "prob_b"]}
    assert probability_names_problem(df, roles, "multiclass", {}, ["M1"]) is not None
    # 메타데이터의 클래스 이름이 있으면 그것과 맞춘다.
    assert probability_names_problem(df, roles, "multiclass", {"detected_classes": ["a", "b"]}, ["M1"]) is None


# ── M4 ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "values,warned",
    [
        ([0.01, 0.2, 0.5], True),       # 초 단위로 보임
        ([12, 25, 40], False),          # ms
        ([0.3, 0.9, 1.0], False),       # 하나라도 1 이상이면 경고 없음
    ],
)
def test_latency_unit_warning(values, warned):
    rows = "\n".join(f"{i},1,1,{v}" for i, v in enumerate(values))
    csv = "id,y_true,y_pred,latency_ms\n" + rows + "\n"
    body = _validate(csv, "binary",
                     [("id", "sample_id"), ("y_true", "y_true"), ("y_pred", "y_pred"), ("latency_ms", "latency")],
                     ["M1"])
    items = _items(body, UNIT)
    if warned:
        assert len(items) == 1
        assert items[0]["status"] == "warning" and items[0]["group"] == "latency"
    else:
        assert items == []


def test_latency_unit_pure_edges():
    assert checks.check_latency_unit(pd.DataFrame({"l": [0.5]}), {}) == []           # 매핑 없음
    assert checks.check_latency_unit(pd.DataFrame({"l": ["x", None]}), {"latency": "l"}) == []  # 숫자 없음
    assert len(checks.check_latency_unit(pd.DataFrame({"l": [0.999]}), {"latency": "l"})) == 1
    assert checks.check_latency_unit(pd.DataFrame({"l": [1.0]}), {"latency": "l"}) == []
