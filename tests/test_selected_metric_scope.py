"""tests/test_selected_metric_scope.py — 결측 제거 대상을 선택 지표로 좁힌다.

ISSUES.md D-06 (2026-09-07 ★확정된 제품 결정 6 의 후속).

종전에는 `ignore` 가 아닌 **모든** 매핑 컬럼의 결측이 행을 버렸다. 그래서 고른 지표가
읽지도 않는 컬럼(샘플 ID, 쓰지 않는 확률 컬럼) 하나에 빈 칸이 있으면 그 행이 평가에서
빠졌다 — 사용자는 자기가 고른 지표와 무관한 이유로 표본을 잃는다.

착수 조건(선행 라운드 실측으로 확정):
  · `preprocess_data`·`engine.evaluate` 두 계층에 인자를 배선해야 한다.
  · `selected_metric_ids=[]` 는 **422 로 거절**한다.
  · 정답·예측 역할은 선택 지표와 무관하게 **항상** dropna 대상이다
    (그래야 `IntCastingNaNError` 로 죽는 경로가 원천 차단된다).
  · 잘못 켜면 D-01(검증↔평가 표본 수 불일치)이 다시 열린다.
"""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.frame import build_evaluation_frame, dropna_roles
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)


def _cm(column, role):
    return ColumnMapping(column=column, role=role)


def _request(metrics, mappings, task=TaskType.binary, **kw):
    return EvaluateRequest(
        task_type=task,
        column_mappings=[_cm(c, r) for c, r in mappings],
        selected_metric_ids=list(metrics),
        metadata=DataMetadata(positive_class="1", negative_class="0", **kw),
    )


def _post(endpoint, csv, request):
    return client.post(
        endpoint,
        files={"file": ("d.csv", csv.encode("utf-8"), "text/csv")},
        data={"data": request.model_dump_json()},
    )


# id 에 빈 칸이 하나 있다. 어떤 지표도 sample_id 를 읽지 않는다.
_CSV_ID_GAP = "i,y,p,s\nA,1,1,0.9\n,1,0,0.2\nC,0,0,0.1\nD,0,1,0.8\n"
_MAPPINGS = [("i", ColumnRole.sample_id), ("y", ColumnRole.y_true),
             ("p", ColumnRole.y_pred), ("s", ColumnRole.score_positive)]


def test_unused_column_gap_no_longer_drops_the_row():
    """[D-06] 아무 지표도 읽지 않는 컬럼의 결측이 표본을 깎지 않는다."""
    body = _post("/api/evaluate", _CSV_ID_GAP, _request(["M1"], _MAPPINGS)).json()
    assert body["dropped_rows"] == 0
    assert body["n_samples"] == 4


def test_column_a_selected_metric_reads_still_drops_the_row():
    """선택 지표가 실제로 읽는 컬럼의 결측은 종전대로 행을 버린다."""
    csv = "i,y,p,s\nA,1,1,0.9\nB,1,0,\nC,0,0,0.1\nD,0,1,0.8\n"
    body = _post("/api/evaluate", csv, _request(["M9"], _MAPPINGS)).json()
    assert body["dropped_rows"] == 1
    assert body["n_samples"] == 3


def test_truth_and_prediction_are_always_dropna_targets():
    """[D-06 착수 조건] 정답·예측은 선택 지표와 무관하게 항상 제외 대상이다.

    M23 은 y_pred 를 읽지 않지만, y_pred 에 NaN 이 남으면 `_coerce_label_types` 의
    `astype(int)` 이 `IntCastingNaNError` 로 죽는다. 그 경로를 원천 차단한다.
    """
    csv = "y,p\n1,1\n1,\n0,0\n"
    resp = _post("/api/evaluate", csv,
                 _request(["M23"], [("y", ColumnRole.y_true), ("p", ColumnRole.y_pred)]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["dropped_rows"] == 1


def test_missing_truth_always_drops_the_row():
    csv = "y,p\n1,1\n,0\n0,0\n"
    body = _post("/api/evaluate", csv,
                 _request(["M23"], [("y", ColumnRole.y_true), ("p", ColumnRole.y_pred)])).json()
    assert body["dropped_rows"] == 1


def test_derivation_source_stays_a_dropna_target():
    """확률에서 예측을 파생할 때 그 확률 컬럼의 결측은 행을 버려야 한다.

    파생에 쓰이는 값이 NaN 이면 예측을 만들 수 없다 — '읽지 않는 컬럼'이 아니다.
    """
    csv = "y,s\n1,0.9\n1,\n0,0.1\n"
    resp = _post("/api/evaluate", csv,
                 _request(["M1"], [("y", ColumnRole.y_true), ("s", ColumnRole.score_positive)]))
    assert resp.status_code == 200, resp.text
    assert resp.json()["dropped_rows"] == 1


def test_latency_gap_never_drops_a_row():
    """지연시간은 부가 측정이다 — 못 잰 샘플을 지표에서 뺄 이유가 없다(종전 동작 유지)."""
    csv = "y,p,l\n1,1,10\n1,0,\n0,0,30\n"
    body = _post("/api/evaluate", csv,
                 _request(["M1"], [("y", ColumnRole.y_true), ("p", ColumnRole.y_pred),
                                   ("l", ColumnRole.latency)])).json()
    assert body["dropped_rows"] == 0


# ── 검증과 평가가 같은 규칙을 쓴다 (D-01 재발 방지) ────────────────────────

def test_validation_and_evaluation_still_agree():
    """[D-01] 좁힌 규칙을 한쪽에만 적용하면 표본 수가 다시 갈라진다."""
    request = _request(["M1"], _MAPPINGS)
    v = _post("/api/validate-data", _CSV_ID_GAP, request).json()
    e = _post("/api/evaluate", _CSV_ID_GAP, request).json()

    summary = {i["label"]: i["value"] for i in v["execution_summary"]}
    assert int(summary["Valid prediction rows"].split()[0]) == e["n_samples"]
    assert int(summary["Excluded samples"].split()[0]) == e["dropped_rows"]


# ── 빈 지표 목록은 422 (착수 조건) ─────────────────────────────────────────

@pytest.mark.parametrize("endpoint", ["/api/evaluate", "/api/validate-data"])
def test_empty_metric_selection_is_rejected(endpoint):
    """지표를 하나도 고르지 않으면 '무엇을 읽는가'가 정의되지 않는다.

    조용히 '전 컬럼'으로 되돌리면 사용자는 좁혀진 줄 알고 넓은 규칙을 받는다.
    """
    payload = {
        "task_type": "binary",
        "column_mappings": [{"column": "y", "role": "y_true"}],
        "selected_metric_ids": [],
        "metadata": {},
    }
    resp = client.post(
        endpoint,
        files={"file": ("d.csv", b"y\n1\n0\n", "text/csv")},
        data={"data": json.dumps(payload)},
    )
    assert resp.status_code == 422, resp.text


# ── dropna_roles 단위 규칙 ─────────────────────────────────────────────────

def test_dropna_roles_excludes_unread_roles():
    roles = dropna_roles("binary", ["M1"], {"sample_id", "y_true", "y_pred", "score_positive", "latency"})
    assert "sample_id" not in roles
    assert "latency" not in roles
    assert {"y_true", "y_pred"} <= roles


def test_dropna_roles_includes_score_when_a_metric_reads_it():
    roles = dropna_roles("binary", ["M9"], {"y_true", "score_positive"})
    assert "score_positive" in roles


def test_dropna_roles_includes_probability_when_prediction_is_derived():
    """예측이 없어 파생해야 하면 확률 컬럼은 '읽는 컬럼'이다."""
    roles = dropna_roles("multilabel", ["M16"], {"true_labels", "score_per_label"})
    assert "score_per_label" in roles


def test_dropna_roles_drops_probability_when_prediction_exists_and_is_unread():
    """하드 예측이 있으면 확률은 파생에 쓰이지 않는다 — 읽지 않는 컬럼이다."""
    roles = dropna_roles("multilabel", ["M16"], {"true_labels", "pred_labels", "score_per_label"})
    assert "score_per_label" not in roles
    assert {"true_labels", "pred_labels"} <= roles


def test_build_evaluation_frame_honours_selected_metrics():
    df = pd.DataFrame({"i": ["A", None], "y": [1, 0], "p": [1, 0]})
    mappings = [{"column": "i", "role": "sample_id"}, {"column": "y", "role": "y_true"},
                {"column": "p", "role": "y_pred"}]

    _, dropped_all, _ = build_evaluation_frame(df.copy(), mappings, "binary", None)
    _, dropped_narrow, _ = build_evaluation_frame(df.copy(), mappings, "binary", ["M1"])

    assert dropped_all == 1, "지표를 모르면 종전대로 보수적으로 전 컬럼을 본다"
    assert dropped_narrow == 0
