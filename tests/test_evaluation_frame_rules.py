"""tests/test_evaluation_frame_rules.py — 평가 대상 행·클래스 규칙 (결정 6 네 갈래).

ISSUES.md C-03 · C-04 · D-04 · D-03 (2026-09-07 ★확정된 제품 결정 6).

① 유령 클래스(C-03) — 예측에만 등장한 클래스는 정답 클래스의 오분류(FN)로 세고,
   클래스 집합은 y_true 기준으로 고정한다. **표본 수는 불변이다.**
② 단일 클래스(C-04) — 클래스가 1종이면 M23 을 '측정 불가'로 표시한다.
③ 라벨 표현형(D-04) — 문자열로 통일하되 전부 숫자면 수치 순서를 유지한다.
④ 약속 문구(D-03) — 구현하지 않은 처리를 handling 에 적지 않는다.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.errors import METRIC_ERROR_KEY
from app.evaluation.labels import (
    normalize_distribution,
    normalize_label,
    parse_label_cell,
    sort_labels,
)
from app.evaluation.metrics import common
from app.evaluation.schemas import EvaluateRequest
from app.main import app

client = TestClient(app)


def _cm(column, role):
    return ColumnMapping(column=column, role=role)


def _evaluate(csv, request):
    return client.post(
        "/api/evaluate",
        files={"file": ("d.csv", csv.encode("utf-8"), "text/csv")},
        data={"data": request.model_dump_json()},
    )


# ── ① 유령 클래스 (C-03) ───────────────────────────────────────────────────
#
# 정답 3클래스(A×3, B×3, C×3), 예측 한 행이 정답에 없는 D.
# D 를 클래스로 편입하면 macro 평균의 분모가 4가 되어 값이 희석된다.
# y_true 로 고정하면 그 행은 A 의 FN 이 되고 분모는 3이다.
_GHOST_CSV = (
    "t,p\n"
    "A,A\nA,A\nA,D\n"
    "B,B\nB,B\nB,B\n"
    "C,C\nC,C\nC,C\n"
)


def _ghost_request(metrics):
    return EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("t", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=list(metrics),
        metadata=DataMetadata(),
    )


def test_ghost_class_does_not_dilute_macro_average():
    """유령 D 가 클래스로 편입되면 macro 분모가 4가 되어 값이 낮아진다.

    y_true 기준 고정: recall = (2/3 + 1 + 1) / 3 = 8/9.
    합집합이면 D 의 recall 0 이 더해져 (2/3+1+1+0)/4 = 2/3 으로 희석된다.
    """
    body = _evaluate(_GHOST_CSV, _ghost_request(["M3"])).json()
    assert body["results"]["success_metrics"]["M3"] == pytest.approx(8 / 9)


def test_ghost_class_is_counted_as_false_negative_of_the_true_class():
    """A 는 3건 중 1건을 놓쳤다 — recall 2/3 로 잡혀야 한다."""
    body = _evaluate(_GHOST_CSV, _ghost_request(["M22"])).json()
    report = body["results"]["success_metrics"]["M22"]
    assert report["A"]["recall"] == pytest.approx(2 / 3)
    assert "D" not in report, "정답에 없는 클래스가 클래스별 표에 행을 만들면 안 된다"


def test_ghost_class_set_agrees_with_imbalance_ratio():
    """M23 은 원래 y_true 만 본다 — 다른 지표의 클래스 집합이 그것과 일치해야 한다."""
    body = _evaluate(_GHOST_CSV, _ghost_request(["M22", "M23"])).json()
    metrics = body["results"]["success_metrics"]
    class_rows = {k for k in metrics["M22"] if k not in
                  ("accuracy", "macro avg", "weighted avg", "micro avg", "samples avg")}
    assert class_rows == set(body["class_distribution"])


def test_ghost_class_keeps_sample_count_unchanged():
    """**표본 수는 불변이다.** 혼동행렬의 합이 평가 행 수와 같아야 한다.

    혼동행렬에서 유령 열을 지우면 그 행이 어느 칸에도 들어가지 않아 합이 줄고,
    성적서 한 문서에 표본 수가 두 개가 된다(4차 라운드가 닫은 B-02 의 재발).
    """
    body = _evaluate(_GHOST_CSV, _ghost_request(["M21"])).json()
    matrix = body["results"]["success_metrics"]["M21"]["matrix"]
    assert sum(sum(row) for row in matrix) == body["n_samples"] == 9


def test_ghost_class_is_warned_about():
    """조용히 처리하지 않는다 — 어떤 클래스가 정답에 없었는지 알린다."""
    body = _evaluate(_GHOST_CSV, _ghost_request(["M3"])).json()
    assert any("D" in w for w in body["warnings"]), body["warnings"]


def test_multilabel_ghost_label_is_excluded_from_the_label_set():
    """multilabel 은 MultiLabelBinarizer 의 fit 대상을 y_true 로 좁혀야 한다."""
    csv = "t,p\na|b,a|z\na,a\nb,b\n"
    req = EvaluateRequest(
        task_type=TaskType.multilabel,
        column_mappings=[_cm("t", ColumnRole.true_labels), _cm("p", ColumnRole.pred_labels)],
        selected_metric_ids=["M22"],
        metadata=DataMetadata(),
    )
    report = _evaluate(csv, req).json()["results"]["success_metrics"]["M22"]
    assert "z" not in report


def test_no_ghost_means_no_behaviour_change():
    """유령이 없으면 아무것도 달라지지 않는다(과잉 변경 방지)."""
    csv = "t,p\nA,A\nA,B\nB,B\nB,B\n"
    body = _evaluate(csv, EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("t", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=["M3", "M1"],
        metadata=DataMetadata(),
    )).json()
    assert body["results"]["success_metrics"]["M1"] == pytest.approx(0.75)
    assert body["results"]["success_metrics"]["M3"] == pytest.approx((1 / 2 + 1) / 2)
    assert body["warnings"] == []


# ── ② 단일 클래스 (C-04) ───────────────────────────────────────────────────

@pytest.mark.parametrize("task, truth_role, csv", [
    (TaskType.multiclass, ColumnRole.y_true, "t,p\nA,A\nA,A\nA,A\n"),
    (TaskType.multilabel, ColumnRole.true_labels, "t,p\na,a\na,a\na,a\n"),
])
def test_single_class_makes_imbalance_ratio_unmeasurable(task, truth_role, csv):
    """클래스가 1종이면 불균형비는 정의되지 않는다.

    종전에는 max/min = 1.0 이 나왔고, 성적서가 그것을 '완벽한 균형'으로 서술했다 —
    실제로는 평가 자체가 성립하지 않는 데이터인데 정반대 결론이 인쇄됐다.
    """
    pred_role = ColumnRole.y_pred if task != TaskType.multilabel else ColumnRole.pred_labels
    body = _evaluate(csv, EvaluateRequest(
        task_type=task,
        column_mappings=[_cm("t", truth_role), _cm("p", pred_role)],
        selected_metric_ids=["M23"],
        metadata=DataMetadata(),
    )).json()
    assert "M23" not in body["results"]["success_metrics"]
    assert "M23" in body["results"]["failed_metrics"]


def test_two_classes_still_computes_imbalance_ratio():
    """두 클래스 이상이면 종전대로 계산한다(과잉 차단 방지)."""
    csv = "t,p\nA,A\nA,A\nA,A\nB,B\n"
    body = _evaluate(csv, EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("t", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=["M23"],
        metadata=DataMetadata(),
    )).json()
    assert body["results"]["success_metrics"]["M23"] == pytest.approx(3.0)


def test_imbalance_ratio_error_is_a_metric_error_not_a_crash():
    """전체 평가를 죽이지 않고 그 지표만 '측정 불가'가 된다."""
    df = pd.DataFrame({"t": ["A", "A", "A"]})
    with pytest.raises(ValueError):
        common.calculate_imbalance_ratio(df, {"y_true": "t"})


# ── ③ 라벨 표현형 (D-04) ───────────────────────────────────────────────────

@pytest.mark.parametrize("cell, expected", [
    ("a|b", ["a", "b"]),
    ("a,b", ["a", "b"]),
    ("['a', 'b']", ["a", "b"]),
    ("[1, 2]", ["1", "2"]),          # literal_eval 이 int 를 만들던 자리
    ("3", ["3"]),
    (["a", 1], ["a", "1"]),
    ("", []),
    (None, []),
    (1.0, ["1"]),                    # float 로 읽힌 정수 라벨
])
def test_label_cell_parses_to_strings(cell, expected):
    """파서는 표기와 무관하게 **항상 문자열** 라벨을 낸다."""
    assert parse_label_cell(cell) == expected


def test_numeric_labels_keep_numeric_order():
    """전부 숫자면 수치 순서 — 단순 문자열 정렬은 1,10,2,3 으로 뒤집는다."""
    assert sort_labels(["10", "2", "1", "3"]) == ["1", "2", "3", "10"]
    assert sort_labels([10, 2, 1, 3]) == ["1", "2", "3", "10"]


def test_non_numeric_labels_fall_back_to_lexicographic_order():
    assert sort_labels(["b", "a", "c"]) == ["a", "b", "c"]
    assert sort_labels(["2", "a", "1"]) == ["1", "2", "a"]


def test_distribution_merge_does_not_lose_counts():
    """`{str(k): v}` 는 int 1 과 str '1' 을 뭉개며 **나중 값이 앞 값을 덮어썼다.**

    실측으로 합 6 이 합 4 가 됐다 — 라벨 2건이 예외 없이 사라진 채 인쇄됐다.
    """
    merged = normalize_distribution({1: 1, 2: 1, "1": 1, "3": 2, "2": 1})
    assert sum(merged.values()) == 6
    assert merged == {"1": 2, "2": 2, "3": 2}


def test_mixed_type_labels_do_not_crash_evaluation():
    """같은 컬럼에 '[1, 2]' 와 '3' 이 섞여도 평가가 끝난다.

    종전에는 MultiLabelBinarizer 가 `'<' not supported between int and str` 로 죽었다.
    """
    csv = 't,p\n"[1, 2]","[1, 2]"\n3,3\n"[1, 3]","[1]"\n'
    body = _evaluate(csv, EvaluateRequest(
        task_type=TaskType.multilabel,
        column_mappings=[_cm("t", ColumnRole.true_labels), _cm("p", ColumnRole.pred_labels)],
        selected_metric_ids=["M16", "M21"],
        metadata=DataMetadata(),
    )).json()
    assert body["results"]["failed_metrics"] == {}
    assert body["results"]["success_metrics"]["M21"]["labels"] == ["1", "2", "3"]


def test_class_distribution_keeps_numeric_order_in_response():
    """성적서 클래스 순서가 1,10,2 로 뒤집히지 않는다."""
    csv = "t,p\n1,1\n2,2\n10,10\n10,10\n"
    body = _evaluate(csv, EvaluateRequest(
        task_type=TaskType.multiclass,
        column_mappings=[_cm("t", ColumnRole.y_true), _cm("p", ColumnRole.y_pred)],
        selected_metric_ids=["M1"],
        metadata=DataMetadata(),
    )).json()
    assert list(body["class_distribution"]) == ["1", "2", "10"]


# ── ④ 약속 문구 (D-03) ─────────────────────────────────────────────────────

_UNIMPLEMENTED_PROMISES = [
    "Keep the first row and exclude later duplicates",
    "Exclude affected rows from evaluation",
]


def test_validation_handling_text_makes_no_unimplemented_promise():
    """검증 표의 '처리 방법'이 실제로 하지 않는 일을 적으면 안 된다.

    `app/` 어디에도 중복 ID 제거(`drop_duplicates`)나 미지 클래스 행 제외(`isin`)가
    없다. 구현하면 표본 수가 줄어 모든 골든과 기존 성적서가 어긋나므로, 결정 6-④ 는
    **구현하지 않고 문구를 사실대로 정정**하는 쪽이다.
    """
    from pathlib import Path

    source = Path("app/analysis/validation_checks.py").read_text(encoding="utf-8")
    offenders = [p for p in _UNIMPLEMENTED_PROMISES if p in source]
    assert offenders == [], f"지키지 않는 약속이 남아 있다: {offenders}"


def test_duplicate_id_handling_says_what_actually_happens():
    csv = "i,t,p\nx,A,A\nx,B,B\ny,A,A\n"
    body = client.post(
        "/api/validate-data",
        files={"file": ("d.csv", csv.encode("utf-8"), "text/csv")},
        data={"data": EvaluateRequest(
            task_type=TaskType.multiclass,
            column_mappings=[
                _cm("i", ColumnRole.sample_id), _cm("t", ColumnRole.y_true),
                _cm("p", ColumnRole.y_pred),
            ],
            selected_metric_ids=["M1"],
            metadata=DataMetadata(),
        ).model_dump_json()},
    ).json()
    item = next(i for i in body["validation_details"] if i["name"] == "Duplicate ID")
    assert item["result"] == "1 rows"
    assert "exclude" not in item["handling"].lower()


def test_metadata_parser_agrees_with_evaluation_parser():
    """[D-04] 분석 단계와 평가 단계가 같은 셀을 같은 라벨 집합으로 읽는다.

    종전에는 `analysis/metadata` 가 '|' 만 쪼개서 `"a,b"` 를 라벨 **1개**로 셌고,
    평가 파서는 라벨 **2개**로 읽었다. metadata 의 산출물은 프론트의
    `detected_labels`·`class_distribution` 이 되어 성적서에 인쇄되므로, 한 문서 안에
    서로 다른 라벨 집합이 실렸다.
    """
    from app.analysis.metadata import extract_metadata

    df = pd.DataFrame({"t": ["a,b", "b", "[1, 2]"], "p": ["a", "b", "1"]})
    mappings = [
        ColumnMapping(column="t", role=ColumnRole.true_labels),
        ColumnMapping(column="p", role=ColumnRole.pred_labels),
    ]
    meta = extract_metadata(TaskType.multilabel, df, df, mappings)

    from_evaluation = sorted({l for cell in df["t"] for l in parse_label_cell(cell)})
    assert sorted(meta.detected_labels) == from_evaluation
    # "a,b" → a,b · "b" → b · "[1, 2]" → 1,2 = 라벨 등장 5회.
    # 종전 파서('|' 만 쪼갬)로는 "a,b" 가 라벨 1개라 3회로 셌다.
    assert sum(meta.class_distribution.values()) == 5
    assert meta.class_distribution == {"1": 1, "2": 1, "a": 1, "b": 2}

    # `column_unique_values` 는 매핑 화면의 값 목록이 된다 — 같은 파서를 써야
    # 사용자가 보는 라벨 후보와 평가가 실제로 읽는 라벨이 같아진다.
    assert meta.column_unique_values["t"] == ["1", "2", "a", "b"]


def test_metadata_class_order_is_numeric_when_labels_are_numeric():
    """[D-04] `sorted(str)` 이 ['1','10','2'] 를 만들던 자리."""
    from app.analysis.metadata import extract_metadata

    df = pd.DataFrame({"t": ["1", "2", "10", "10"], "p": ["1", "2", "10", "10"]})
    meta = extract_metadata(
        TaskType.multiclass,
        df,
        df,
        [
            ColumnMapping(column="t", role=ColumnRole.y_true),
            ColumnMapping(column="p", role=ColumnRole.y_pred),
        ],
    )
    assert meta.detected_classes == ["1", "2", "10"]
