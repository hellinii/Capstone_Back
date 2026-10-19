"""클래스 표기 한 곳(normalize_label) — 빈 칸 섞인 숫자 정답·앞뒤 공백 클래스.

정답 컬럼이 숫자인데 빈 칸이 하나라도 있으면 pandas 가 float 으로 읽어 값이 1.0·0.0 이 된다. 화면에 보이는
고유값 목록(column_unique_values)은 normalize_label 을 거쳐 '1'·'0' 인데, 종전에는
- 이진: metadata.positive_class 가 str(1.0) = '1.0' 이라 프론트가 '목록에 없는 양성'으로 보고 비웠고,
  사용자가 고른 '0' 은 예측 파생이 str 비교로 못 찾아 기본 양성(1)으로 바꿔 버렸다(지표 계산은 0 을 양성으로 봄).
- 다중: 확률 컬럼 prob_0 의 '0' 과 클래스 '0.0' 이 맞지 않아 검증·평가가 400, metadata 가 있으면 파생 예측이
  문자열 '0' 이 되어 지표가 'Mix of label input types' 로 실패했다.
- 앞뒤 공백 클래스(' cat')는 파생 예측이 'cat' 이 되어 정답 ' cat' 과 어긋나 M1 이 조용히 0 이 됐다.
이제 클래스 이름은 normalize_label 로 맞추고, 원래 값(dtype·공백 그대로)은 labels.find_label 로 되찾는다.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analysis.metadata import extract_metadata
from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.labels import find_label
from app.evaluation.metrics.binary import _resolve_positive_value
from app.evaluation.metrics.common import _get_binary_kwargs
from app.evaluation.preprocessor import _known_targets
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)

NAME_CHECK = "Probability column names"

# 빈 칸 한 줄(3번째 데이터 행) 때문에 y_true 가 float 이 된다.
BIN_CSV = "y_true,score\n1,0.9\n0,0.2\n,0.7\n1,0.4\n0,0.1\n1,0.8\n"
BIN_MAPS = [("y_true", "y_true"), ("score", "score_positive")]

MC_CSV = (
    "y_true,prob_0,prob_1,prob_2\n0,0.8,0.1,0.1\n1,0.1,0.8,0.1\n2,0.1,0.1,0.8\n"
    ",0.3,0.3,0.4\n1,0.6,0.3,0.1\n2,0.2,0.2,0.6\n"
)
MC_MAPS = [("y_true", "y_true"), ("prob_0", "prob_per_class"), ("prob_1", "prob_per_class"),
           ("prob_2", "prob_per_class")]

SPACE_CSV = "y_true,prob_cat,prob_dog\n cat,0.9,0.1\n dog,0.2,0.8\n cat,0.4,0.6\n dog,0.3,0.7\n"
SPACE_MAPS = [("y_true", "y_true"), ("prob_cat", "prob_per_class"), ("prob_dog", "prob_per_class")]


def _mappings(maps):
    return [ColumnMapping(column=c, role=ColumnRole(r)) for c, r in maps]


def _metadata(csv: str, task: str, maps) -> DataMetadata:
    df = pd.read_csv(io.StringIO(csv))
    return extract_metadata(TaskType(task), df, df.head(30), _mappings(maps))


def _post(endpoint: str, csv: str, task: str, maps, metrics, metadata: DataMetadata):
    req = EvaluateRequest(task_type=TaskType(task), column_mappings=_mappings(maps),
                          selected_metric_ids=metrics, metadata=metadata)
    return client.post(endpoint, files={"file": ("d.csv", csv.encode(), "text/csv")},
                       data={"data": req.model_dump_json()})


def _name_errors(csv, task, maps, metadata):
    r = _post("/api/validate-data", csv, task, maps, ["M1"], metadata)
    assert r.status_code == 200, r.text
    return [d for d in r.json()["validation_details"] if d["name"] == NAME_CHECK]


# ── 이진: metadata 의 양성·음성 표기 ───────────────────────────────────────────

def test_binary_metadata_classes_match_unique_values_when_blank_makes_float():
    md = _metadata(BIN_CSV, "binary", BIN_MAPS)
    assert (md.positive_class, md.negative_class) == ("1", "0")
    shown = md.column_unique_values["y_true"]
    assert shown == ["0", "1"]
    assert md.positive_class in shown and md.negative_class in shown


def test_binary_metadata_strips_spaces_like_unique_values():
    md = _metadata("y_true,score\n yes,0.9\n no,0.2\n yes,0.6\n", "binary", BIN_MAPS)
    assert (md.positive_class, md.negative_class) == ("yes", "no")
    assert md.positive_class in md.column_unique_values["y_true"]


def test_binary_user_picked_negative_label_is_honoured_by_derivation_and_metrics():
    """사용자가 '0' 을 양성으로 고르면 파생·지표가 모두 0 을 양성으로 본다(종전: 파생만 1 로 바꿈 → M1 0.8)."""
    md = _metadata(BIN_CSV, "binary", BIN_MAPS)
    md.positive_class, md.negative_class = "0", "1"
    r = _post("/api/evaluate", BIN_CSV, "binary", BIN_MAPS, ["M1", "M2"], md)
    assert r.status_code == 200, r.text
    res = r.json()["results"]["success_metrics"]
    assert res["derived_prediction"]["positive_class"] == "0"
    assert res["derived_prediction"]["negative_class"] == "1"
    # 점수 ≥0.5 → 0, 아니면 1. 남는 5행 중 맞는 것은 (1, 0.4) 하나.
    assert res["M1"] == pytest.approx(0.2)
    assert res["confusion_counts"]["positive_class"] == "0"


def test_binary_default_positive_reports_plain_label():
    md = _metadata(BIN_CSV, "binary", BIN_MAPS)
    r = _post("/api/evaluate", BIN_CSV, "binary", BIN_MAPS, ["M1"], md)
    assert r.status_code == 200, r.text
    res = r.json()["results"]["success_metrics"]
    assert res["derived_prediction"]["positive_class"] == "1"
    assert res["confusion_counts"]["positive_class"] == "1"
    assert res["M1"] == pytest.approx(0.8)


# ── 다중: 확률 컬럼 이름과 클래스 ─────────────────────────────────────────────

@pytest.mark.parametrize("with_metadata", [True, False])
def test_multiclass_blank_int_truth_resolves_prob_names(with_metadata):
    md = _metadata(MC_CSV, "multiclass", MC_MAPS) if with_metadata else DataMetadata()
    assert _name_errors(MC_CSV, "multiclass", MC_MAPS, md) == []
    r = _post("/api/evaluate", MC_CSV, "multiclass", MC_MAPS, ["M1", "M2"], md)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert results["failed_metrics"] == {}
    # 빈 칸 행을 뺀 5행 중 (1 → argmax 0) 하나만 틀림.
    assert results["success_metrics"]["M1"] == pytest.approx(0.8)
    assert results["success_metrics"]["derived_prediction"]["class_order"] == ["0", "1", "2"]


@pytest.mark.parametrize("with_metadata", [True, False])
def test_multiclass_spaced_classes_resolve_and_score_correctly(with_metadata):
    md = _metadata(SPACE_CSV, "multiclass", SPACE_MAPS) if with_metadata else DataMetadata()
    assert _name_errors(SPACE_CSV, "multiclass", SPACE_MAPS, md) == []
    r = _post("/api/evaluate", SPACE_CSV, "multiclass", SPACE_MAPS, ["M1"], md)
    assert r.status_code == 200, r.text
    res = r.json()["results"]["success_metrics"]
    assert res["M1"] == pytest.approx(0.75)  # 종전 0.0 — 파생 예측 'cat' ≠ 정답 ' cat'
    assert len(res["confusion_counts"]["rows"]) == 2  # 유령 클래스 행(cat·dog)이 덧붙지 않는다


def test_known_targets_normalizes_both_sources():
    df = pd.DataFrame({"t": [0.0, 1.0, np.nan, 2.0]})
    assert _known_targets("multiclass", df, "t", {}) == ["0", "1", "2"]
    assert _known_targets("multiclass", df, "t", {"detected_classes": [0.0, " 1"]}) == ["0", "1"]


# ── 원래 값 되찾기(find_label) ─────────────────────────────────────────────────

def test_find_label_returns_original_value():
    assert find_label([0.0, 1.0], "1") == 1.0
    assert isinstance(find_label(np.array([0.0, 1.0]), "0"), np.floating)
    assert find_label([" cat", " dog"], "dog") == " dog"
    assert find_label([1, 0], 1) == 1
    assert find_label(["a", "b"], "c") is None
    assert find_label([], "a") is None
    assert find_label(["a"], None) is None


def test_positive_value_resolves_against_float_and_spaced_truth():
    assert _resolve_positive_value(pd.Series([0.0, 1.0, 1.0]), "1") == 1.0
    assert _resolve_positive_value(pd.Series([" cat", " dog"]), "dog") == " dog"
    assert _get_binary_kwargs(pd.Series([" cat", " dog"]), {"_positive_class": "cat"})["pos_label"] == " cat"
    # 기존 정상 데이터는 종전과 같은 값
    assert _resolve_positive_value(pd.Series([0, 1, 1]), "1") == 1
    assert _resolve_positive_value(pd.Series(["no", "yes"]), "yes") == "yes"
    assert _get_binary_kwargs(pd.Series([0, 1]), {"_positive_class": "0"})["pos_label"] == 0
