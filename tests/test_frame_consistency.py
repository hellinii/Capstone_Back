"""검증과 평가가 같은 데이터를 같은 규칙으로 판단하는지 검증.

ISSUES.md D-01 — 한 시스템에 **결측 판정 기준이 셋** 있다.

  (A) 표시용   validation_checks.check_missing_values — 매핑된 비-ignore 컬럼 중
                latency 만 빼고 NaN 이 하나라도 있는 행을 센다.
  (B) 표본수용 validation_service — `df_work.dropna()`. 대상은 매핑된 비-ignore **전** 컬럼
                (latency 포함). 이 값이 성적서 6절의 "유효 예측 건수"로 인쇄된다.
  (C) 평가용   preprocessor — multilabel 결측은 ''로 채워 **살리고**, dropna 에서 latency 를
                **제외**한다. 이 값으로 실제 지표가 계산된다.

(B)≠(C) 이므로 **성적서 6절에 인쇄되는 표본 수가 지표를 만든 표본 수와 다르다.**

D-02 는 그 파생이다 — 후속 검사(중복 ID·클래스 불일치·task 별 검사)가 과다 축소된
프레임 위에서 돌아 허위 경고를 만든다.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from router_cases import CASES, request_json

client = TestClient(app)


def _payload(task: str, mappings: list[dict], metric_ids: list[str], positive_class: str = "1") -> str:
    """실제 EvaluateRequest 계약에 맞는 최소 페이로드."""
    return json.dumps({
        "task_type": task,
        "column_mappings": [{**m, "sample_values": []} for m in mappings],
        "selected_metric_ids": metric_ids,
        "metadata": {
            "positive_class": positive_class,
            "negative_class": "0",
            "positive_class_ambiguous": False,
            "detected_classes": [],
            "detected_labels": [],
            "class_distribution": {},
            "column_unique_values": {},
        },
        "beta": 1.0,
    })


def _summary(body: dict) -> dict[str, str]:
    return {item["label"]: item["value"] for item in body["execution_summary"]}


def _rows(value: str) -> int:
    """'198 rows' → 198"""
    return int(value.split()[0])


def _validate(task: str, key: str):
    path = CASES[task][key]
    return client.post(
        "/api/validate-data",
        files={"file": (path.name, path.read_bytes(), "text/csv")},
        data={"data": request_json(task)},
    )


def _evaluate(task: str, key: str):
    path = CASES[task][key]
    return client.post(
        "/api/evaluate",
        files={"file": (path.name, path.read_bytes(), "text/csv")},
        data={"data": request_json(task)},
    )


# ── D-01: 검증이 인쇄하는 표본 수 == 평가가 실제로 쓴 표본 수 ────────────────────

@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_clean_dataset_sample_counts_agree(task):
    """[D-01] 결측이 없는 데이터셋에서는 원래도 일치한다(회귀 방지)."""
    v = _validate(task, "csv").json()
    e = _evaluate(task, "csv").json()

    s = _summary(v)
    total = _rows(s["Total validated rows"])
    assert _rows(s["Valid prediction rows"]) == total - e["dropped_rows"]


def test_multilabel_missing_values_sample_counts_agree():
    """[D-01] 결측이 섞인 데이터셋에서 두 숫자가 어긋나던 것을 고정한다.

    검증이 성적서 6절에 인쇄하는 "유효 예측 건수"와 평가가 실제로 쓴 행 수가 같아야 한다.
    (저장소의 multilabel_200_errors.csv 는 점수 범위 위반도 함께 갖고 있어 이제 평가가
     정당하게 거절하므로 — 아래 D-09 테스트 참조 — 결측만 있는 데이터로 이 성질을 고정한다.)
    """
    csv = (
        "id,true_labels,pred_labels,score_a\n"
        "1,a|b,a|b,0.9\n"
        "2,a,a,\n"          # score_a 결측 → 행 제외
        "3,b,b,0.4\n"
    )
    data = _payload(
        "multilabel",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "true_labels", "role": "true_labels"},
            {"column": "pred_labels", "role": "pred_labels"},
            {"column": "score_a", "role": "score_per_label"},
        ],
        ["M16"],
    )
    files = {"file": ("ml.csv", csv.encode("utf-8"), "text/csv")}
    v = client.post("/api/validate-data", files=files, data={"data": data}).json()
    e = client.post("/api/evaluate", files=files, data={"data": data}).json()

    s = _summary(v)
    total = _rows(s["Total validated rows"])
    eval_rows = total - e["dropped_rows"]

    assert _rows(s["Valid prediction rows"]) == eval_rows, (
        f"성적서 6절은 {s['Valid prediction rows']} 라고 인쇄하는데 "
        f"지표는 {eval_rows}행으로 계산됐다"
    )
    assert _rows(s["Excluded samples"]) == e["dropped_rows"]


def test_out_of_range_scores_are_refused_by_both_layers():
    """[D-09·D-08] 검증이 통과시킨 데이터를 평가가 거절하면 안 된다.

    종전 multilabel 검증에는 점수 범위 검사가 아예 없어서 `/api/validate-data` 는
    errors 0 을 돌려주고 `/api/evaluate` 만 400 을 냈다. 프론트 게이트는 error_count 만
    보므로 사용자는 6단계를 다 지난 뒤 성적서 직전에 막혔다.

    게다가 평가 쪽 검사는 역할당 마지막 컬럼 하나만 봐서(D-08), 같은 파일이 매핑 순서에
    따라 200/400 으로 갈렸다. 두 결함이 함께 닫혔는지 실제 픽스처로 확인한다.
    """
    v = _validate("multilabel", "csv_errors").json()
    e = _evaluate("multilabel", "csv_errors")

    score_items = [i for i in v["validation_details"] if i["name"] == "Score range error"]
    assert score_items, "multilabel 검증이 점수 범위를 아예 검사하지 않는다"
    assert score_items[0]["status"] == "error"
    assert v["error_count"] >= 1, "검증이 통과시키면 프론트 게이트가 사용자를 통과시킨다"
    assert e.status_code == 400, "검증이 error 라고 한 데이터를 평가가 받아들이면 안 된다"


def test_multilabel_empty_labels_are_not_counted_as_missing():
    """[D-01] 빈 멀티레이블 셀은 '해당 레이블 없음'이라는 정상 입력이다.

    평가 전처리는 이것을 ''로 채워 살린다. 검증만 결측으로 세면 표본 수가 어긋난다.
    """
    csv = "id,true_labels,pred_labels\n1,a|b,a|b\n2,,a\n3,c,c\n"
    data = _payload(
        "multilabel",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "true_labels", "role": "true_labels"},
            {"column": "pred_labels", "role": "pred_labels"},
        ],
        ["M16"],
    )

    resp = client.post(
        "/api/validate-data",
        files={"file": ("ml.csv", csv.encode(), "text/csv")},
        data={"data": data},
    )
    assert resp.status_code == 200, resp.text

    s = _summary(resp.json())
    assert _rows(s["Valid prediction rows"]) == 3, (
        "빈 레이블 행이 결측으로 제외됐다 — 평가는 이 행을 살린다"
    )


# ── 재료 선택 점검: 검증이 미리 보는 분포 == 성적서 3절이 쓰는 평가 분포 ────────────

@pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
def test_validate_class_distribution_equals_evaluate(task):
    """검증 응답의 class_distribution 은 평가 응답과 같아야 한다.

    재료 선택 점검(4번 줄: IR > 9 이면 AUPRC 필요)은 검증 화면에서 이 분포로 IR 을 셈한다.
    성적서 3절은 평가 응답의 분포로 IR 을 찍는다. 둘이 어긋나면 화면은 "불균형 큼"이라
    경고하는데 성적서는 IR 8.9 를 찍는 식의 모순이 생긴다.
    """
    v = _validate(task, "csv").json()
    e = _evaluate(task, "csv").json()
    assert e["class_distribution"], "평가 분포가 비어 있으면 비교가 무의미하다"
    assert v["class_distribution"] == e["class_distribution"]
    assert list(v["class_distribution"]) == list(e["class_distribution"]), "클래스 순서도 같아야 한다"


def test_validate_class_distribution_parses_multilabel_cells_like_evaluate():
    """멀티레이블 원문 셀("a|b", "a,b", 빈 셀)을 평가와 같은 파서로 센다."""
    csv = "id,true_labels,pred_labels\n1,a|b,a\n2,,a\n3,\"a,c\",c\n4,10,10\n5,2,2\n"
    data = _payload(
        "multilabel",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "true_labels", "role": "true_labels"},
            {"column": "pred_labels", "role": "pred_labels"},
        ],
        ["M16"],
    )
    files = {"file": ("ml.csv", csv.encode(), "text/csv")}
    v = client.post("/api/validate-data", files=files, data={"data": data}).json()
    e = client.post("/api/evaluate", files=files, data={"data": data}).json()
    assert v["class_distribution"] == {"10": 1, "2": 1, "a": 2, "b": 1, "c": 1}
    assert v["class_distribution"] == e["class_distribution"]


def test_validate_without_frame_has_no_class_distribution():
    """필수 컬럼이 없어 프레임을 만들지 못하면 분포를 지어내지 않는다(null)."""
    csv = "id,t\n1,0\n2,1\n"
    data = _payload(
        "binary",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "t", "role": "y_true"},
            {"column": "p", "role": "y_pred"},
        ],
        ["M1"],
    )
    body = client.post(
        "/api/validate-data",
        files={"file": ("x.csv", csv.encode(), "text/csv")},
        data={"data": data},
    ).json()
    assert body["error_count"] >= 1
    assert body["class_distribution"] is None


# ── D-02: 후속 검사가 과다 축소된 프레임 위에서 허위 경고를 만들지 않는다 ────────

def test_latency_missing_does_not_shrink_class_checks():
    """[D-02] latency 결측이 클래스 검사용 프레임을 줄여 허위 경고를 만들면 안 된다.

    재현되던 상태 — 정상 binary 6행인데 latency 절반이 결측:
      검증 → "Class mismatch: Pred has unknown classes: 1",
             "Binary class system error: Expected 2 classes, found 1"
      평가 → 6행 정상 처리, class_distribution={0:4, 1:2}
    데이터에는 아무 문제가 없다.
    """
    csv = "id,t,p,ms\n1,0,0,10\n2,0,0,20\n3,0,0,30\n4,0,1,40\n5,1,1,\n6,1,1,\n"
    data = _payload(
        "binary",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "t", "role": "y_true"},
            {"column": "p", "role": "y_pred"},
            {"column": "ms", "role": "latency"},
        ],
        ["M1"],
    )

    body = client.post(
        "/api/validate-data",
        files={"file": ("lat.csv", csv.encode(), "text/csv")},
        data={"data": data},
    ).json()

    names = [d["name"] for d in body["validation_details"] if d["status"] in ("warning", "error")]
    assert not any("Class mismatch" in n for n in names), f"허위 클래스 경고: {names}"
    assert not any("Binary class system error" in n for n in names), f"허위 클래스 경고: {names}"

    s = _summary(body)
    assert _rows(s["Valid prediction rows"]) == 6, "latency 결측이 평가 표본을 줄이면 안 된다"


# ── 프레임 헬퍼 단위 ───────────────────────────────────────────────────────────

def test_required_columns_preserves_input_order():
    """[D-17] 컬럼 순서가 결정적이어야 한다.

    종전 `list(set([...]))` 는 순서가 비결정적이라 오류 메시지와 프레임 컬럼 순서가
    실행마다 달라졌다.
    """
    from app.evaluation.frame import required_columns

    mappings = [
        {"column": "id", "role": "sample_id"},
        {"column": "t", "role": "y_true"},
        {"column": "junk", "role": "ignore"},
        {"column": "p", "role": "y_pred"},
        {"column": "t", "role": "y_true"},  # 중복 매핑
    ]
    assert required_columns(mappings) == ["id", "t", "p"]


def test_build_frame_keeps_latency_missing_rows():
    """[D-01] latency 결측은 행을 버리는 사유가 아니다 — 부가 측정이다."""
    import pandas as pd

    from app.evaluation.frame import build_evaluation_frame

    df = pd.DataFrame({"t": [0, 1, 0], "p": [0, 1, 1], "ms": [10, None, 30]})
    mappings = [
        {"column": "t", "role": "y_true"},
        {"column": "p", "role": "y_pred"},
        {"column": "ms", "role": "latency"},
    ]

    frame, dropped, notes = build_evaluation_frame(df, mappings, "binary")

    assert len(frame) == 3
    assert dropped == 0
    assert notes == []


def test_build_frame_drops_rows_missing_a_scored_column():
    """[D-01] 지표가 읽는 컬럼의 결측은 행을 버린다(기존 동작 보존)."""
    import pandas as pd

    from app.evaluation.frame import build_evaluation_frame

    df = pd.DataFrame({"t": [0, 1, 0], "p": [0, None, 1]})
    mappings = [{"column": "t", "role": "y_true"}, {"column": "p", "role": "y_pred"}]

    frame, dropped, notes = build_evaluation_frame(df, mappings, "binary")

    assert len(frame) == 2
    assert dropped == 1
    assert "1개 행이 결측치" in notes[0]


def test_build_frame_raises_on_missing_required_column():
    """[D-01] 매핑된 컬럼이 데이터에 없으면 명확히 실패한다(기존 동작 보존)."""
    import pandas as pd
    import pytest as _pytest

    from app.evaluation.frame import build_evaluation_frame

    df = pd.DataFrame({"t": [0, 1]})
    mappings = [{"column": "t", "role": "y_true"}, {"column": "없는컬럼", "role": "y_pred"}]

    with _pytest.raises(ValueError, match="없는컬럼"):
        build_evaluation_frame(df, mappings, "binary")


# ── S10·S2: Class mismatch 는 평가와 같은 표기(normalize_label)로 비교하고, 표기 차이는 따로 알린다 ────────

def test_blank_true_cell_does_not_raise_false_class_mismatch():
    """[S10] 정답 칸 하나가 비어 y_true 가 float(0.0·1.0), y_pred 가 int(0·1)로 읽히면
    종전엔 "Pred has unknown classes: 0, 1" 이라는 틀린 경고가 났다(평가는 정상 계산)."""
    csv = "id,t,p\n1,0,0\n2,0,1\n3,1,1\n4,1,0\n5,,1\n6,0,0\n"
    data = _payload(
        "binary",
        [
            {"column": "id", "role": "sample_id"},
            {"column": "t", "role": "y_true"},
            {"column": "p", "role": "y_pred"},
        ],
        ["M1"],
    )
    body = client.post(
        "/api/validate-data",
        files={"file": ("blank.csv", csv.encode(), "text/csv")},
        data={"data": data},
    ).json()
    item = next(d for d in body["validation_details"] if d["name"] == "Class mismatch")
    assert item["status"] == "pass", item
    assert item["result"] == "None"
    notation = next(d for d in body["validation_details"] if d["name"] == "Label notation")
    assert notation["status"] == "pass", notation


def test_class_mismatch_unit_numeric_spelling_and_whitespace():
    """표기 비교 규칙을 함수 단위로 고정한다: 1.0↔1, ' cat'↔'cat' 은 같은 클래스다(S2 — 평가가 같은 표기로
    맞춘다). 종전에는 공백 차이를 '예측에 미정의 클래스'로 알렸는데, 화면(HTML)에서 앞 공백이 안 보여 '있는
    클래스가 미정의'라는 문장이 됐고 평가는 실제로 둘을 다른 클래스로 셌다."""
    import pandas as _pd
    from app.analysis.validation_checks import check_class_mismatch, check_label_notation

    mapping = {"y_true": "t", "y_pred": "p"}
    same = check_class_mismatch(_pd.DataFrame({"t": [0.0, 1.0, 1.0], "p": [0, 1, 0]}), mapping)[0]
    assert same.status == "pass" and same.result == "None"

    # 정수 예측이 float 정답에 없는 진짜 미지 클래스(2)는 계속 알린다 — 표기는 '2'
    extra = check_class_mismatch(_pd.DataFrame({"t": [0.0, 1.0], "p": [0, 2]}), mapping)[0]
    assert extra.status == "warning"
    assert extra.result == "Pred has unknown classes: 2"

    # 앞뒤 공백 차이는 미지 클래스가 아니다 — 대신 표기 맞춤 항목이 무엇을 합쳤는지 말한다
    for t, p, expected in (
        ([" cat", "dog"], ["cat", "dog"], "정답 ' cat' → 'cat'"),
        (["cat", "dog"], [" cat", "dog"], "예측 ' cat' → 'cat'"),
    ):
        df = _pd.DataFrame({"t": t, "p": p})
        assert check_class_mismatch(df, mapping)[0].status == "pass"
        notation = check_label_notation(df, mapping)[0]
        assert notation.status == "warning"
        assert expected in notation.result

    # 소수 라벨(0.5)은 정수로 뭉개지 않는다
    frac = check_class_mismatch(_pd.DataFrame({"t": [0.0, 1.0], "p": [0.5, 1.0]}), mapping)[0]
    assert frac.status == "warning" and frac.result == "Pred has unknown classes: 0.5"
