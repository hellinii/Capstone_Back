"""규칙 폴백 매퍼의 이름 규칙 — M8(식별자 과검출·멀티 이름·pos 오검출)·M4(지연시간 이름).

OpenAI 키가 없거나 LLM 이 실패하면 자동 매핑은 컬럼 이름만 보고 역할을 고른다. 종전에는 부분 문자열로
봐서 valid_flag·width 를 샘플 ID 로, position 을 양성 확률로 잡았고, true_labels·latency_ms 는 몰랐다.
"""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.analysis.fallback_mapper import analyze_columns_fallback, fallback_role, name_words
from app.core.schemas import ColumnRole, TaskType, VALID_ROLES_BY_TASK

B, MC, ML = TaskType.binary, TaskType.multiclass, TaskType.multilabel


def _roles(columns, task_type=B):
    df = pd.DataFrame({c: [1, 0] for c in columns})
    resp = analyze_columns_fallback(task_type=task_type, columns=columns, df=df)
    return {m.column: m.role for m in resp.column_mappings}, resp


def test_name_words_splits_separators_and_camel_case():
    assert name_words("imageId") == ["image", "id"]
    assert name_words("row-index") == ["row", "index"]
    assert name_words("Sample ID") == ["sample", "id"]
    assert name_words("latency.ms") == ["latency", "ms"]


@pytest.mark.parametrize(
    "column,expected",
    [
        ("id", ColumnRole.sample_id),
        ("ID", ColumnRole.sample_id),
        ("image_id", ColumnRole.sample_id),
        ("imageId", ColumnRole.sample_id),
        ("sample_id", ColumnRole.sample_id),
        ("row_index", ColumnRole.sample_id),
        ("idx", ColumnRole.sample_id),
        ("uid", ColumnRole.sample_id),
        # M8: 이름에 'id' 가 '들어 있기만' 한 컬럼은 식별자가 아니다.
        ("valid_flag", ColumnRole.ignore),
        ("width", ColumnRole.ignore),
        ("hidden_size", ColumnRole.ignore),
        ("video", ColumnRole.ignore),
        # M8+: 'pos' 가 든 낱말은 양성 확률이 아니다.
        ("position", ColumnRole.ignore),
        ("pos_x", ColumnRole.ignore),
    ],
)
def test_identifier_and_false_positive_rules(column, expected):
    assert fallback_role(B, column) == expected


@pytest.mark.parametrize(
    "column", ["latency", "latency_ms", "inference_time", "inference_time_ms",
               "response_time", "response_time_ms", "elapsed_ms", "latencyMs", "avg_latency"],
)
@pytest.mark.parametrize("task_type", list(TaskType))
def test_latency_names_map_to_latency(column, task_type):
    """M4: 지연시간 이름 — 세 유형 모두에서 latency 역할."""
    assert fallback_role(task_type, column) == ColumnRole.latency


def test_elapsed_without_ms_is_not_guessed():
    """단위가 없는 elapsed 는 초일 수도 있어 잡지 않는다 — 설계 목록에 있는 이름만."""
    assert fallback_role(B, "elapsed") == ColumnRole.ignore


@pytest.mark.parametrize(
    "column,truth,pred",
    [("true_labels", True, False), ("y_true", True, False), ("label", True, False),
     ("pred_labels", False, True), ("y_pred", False, True), ("prediction", False, True)],
)
def test_truth_and_prediction_names_per_task(column, truth, pred):
    """M8+: 멀티레이블 이름(true_labels·pred_labels)도 알아본다. 유형마다 맞는 역할로 간다."""
    if truth:
        assert fallback_role(ML, column) == ColumnRole.true_labels
        assert fallback_role(B, column) == ColumnRole.y_true
        assert fallback_role(MC, column) == ColumnRole.y_true
    if pred:
        assert fallback_role(ML, column) == ColumnRole.pred_labels
        assert fallback_role(B, column) == ColumnRole.y_pred


@pytest.mark.parametrize(
    "task_type,column,expected",
    [
        (B, "score", ColumnRole.score_positive),
        (B, "prob_1", ColumnRole.score_positive),
        (B, "positive_score", ColumnRole.score_positive),
        (MC, "prob_cat", ColumnRole.prob_per_class),
        (MC, "probability_dog", ColumnRole.prob_per_class),
        (MC, "p_cat", ColumnRole.prob_per_class),
        (ML, "score_sports", ColumnRole.score_per_label),
        (ML, "prob_news", ColumnRole.score_per_label),
    ],
)
def test_probability_names(task_type, column, expected):
    assert fallback_role(task_type, column) == expected


def test_second_column_of_a_single_role_is_ignored_with_notice():
    """M8: id + row_index 이면 첫째만 sample_id. 둘째는 ignore 로 내리고 안내한다(확정이 막히지 않게)."""
    roles, resp = _roles(["id", "row_index", "y_true", "y_pred"])
    assert roles["id"] == ColumnRole.sample_id
    assert roles["row_index"] == ColumnRole.ignore
    notes = [n for n in resp.column_notes if n.llm_column == "row_index"]
    assert len(notes) == 1
    assert notes[0].status == "duplicate_role"
    assert "'id'" in notes[0].message and "매핑 화면" in notes[0].message


def test_probability_roles_may_take_many_columns():
    roles, resp = _roles(["y_true", "prob_cat", "prob_dog"], MC)
    assert roles["prob_cat"] == roles["prob_dog"] == ColumnRole.prob_per_class
    assert resp.column_notes == []


@pytest.mark.parametrize("task_type", list(TaskType))
def test_fallback_roles_are_always_allowed(task_type):
    cols = ["id", "true_labels", "pred_labels", "y_true", "y_pred", "prob_a", "score_b", "latency_ms", "valid_flag"]
    roles, _ = _roles(cols, task_type)
    assert roles
    allowed = set(VALID_ROLES_BY_TASK[task_type])
    assert {c: r for c, r in roles.items() if r not in allowed} == {}


# ── 끝단: OpenAI 키 없이 /api/analyze-columns 가 규칙 폴백으로 매핑한다 ─────────────

def _analyze_without_key(csv: bytes, task_type: str):
    from app import main

    with TestClient(main.app) as c:
        main.app.state.openai_client = None  # 키 없음 → 규칙 폴백
        return c.post(
            "/api/analyze-columns",
            data={"task_type": task_type},
            files={"file": ("data.csv", io.BytesIO(csv), "text/csv")},
        )


def test_no_key_binary_upload_maps_valid_flag_latency_and_id():
    csv = b"image_id,valid_flag,y_true,y_pred,latency_ms\n1,1,0,0,12.5\n2,0,1,1,30.1\n3,1,1,0,18.0\n"
    r = _analyze_without_key(csv, "binary")
    assert r.status_code == 200, r.text
    roles = {m["column"]: m["role"] for m in r.json()["column_mappings"]}
    assert roles == {
        "image_id": "sample_id",
        "valid_flag": "ignore",
        "y_true": "y_true",
        "y_pred": "y_pred",
        "latency_ms": "latency",
    }


def test_no_key_multilabel_upload_maps_true_and_pred_labels():
    csv = b"id,true_labels,pred_labels,score_sports,score_news,inference_time_ms\n" \
          b"1,sports|news,sports,0.9,0.4,20\n2,news,news,0.1,0.8,22\n"
    r = _analyze_without_key(csv, "multilabel")
    assert r.status_code == 200, r.text
    roles = {m["column"]: m["role"] for m in r.json()["column_mappings"]}
    assert roles == {
        "id": "sample_id",
        "true_labels": "true_labels",
        "pred_labels": "pred_labels",
        "score_sports": "score_per_label",
        "score_news": "score_per_label",
        "inference_time_ms": "latency",
    }


# ── S9 검토: 종전 규칙이 잡던 흔한 이름을 놓치지 않는다 ─────────────────────────────

@pytest.mark.parametrize("column", ["sampleid", "rowid", "userid", "fileid", "imageid", "SampleID", "RowId"])
def test_compound_identifier_names_stay_sample_id(column):
    """구분자 없이 붙여 쓴 식별자 — 종전 부분 문자열 규칙이 sample_id 로 잡던 꼴."""
    assert fallback_role(B, column) == ColumnRole.sample_id


@pytest.mark.parametrize("column", ["valid", "grid", "void", "solid", "paid"])
def test_words_ending_in_id_are_not_identifiers(column):
    assert fallback_role(B, column) == ColumnRole.ignore


@pytest.mark.parametrize(
    "task_type,column,expected",
    [
        (B, "probs", ColumnRole.score_positive),
        (B, "scores", ColumnRole.score_positive),
        (B, "prob1", ColumnRole.score_positive),
        (B, "y_probas", ColumnRole.score_positive),
        (ML, "scores_sports", ColumnRole.score_per_label),
        (MC, "probabilities_cat", ColumnRole.prob_per_class),
        (B, "probe", ColumnRole.ignore),
    ],
)
def test_plural_and_numbered_probability_words(task_type, column, expected):
    assert fallback_role(task_type, column) == expected


@pytest.mark.parametrize(
    "truth,columns,expected",
    [
        ([0, 1, 0, 1], ["prob_0", "prob_1"], "prob_1"),
        ([0, 1, 0, 1], ["prob_1", "prob_0"], "prob_1"),
        (["cat", "dog", "cat", "dog"], ["p_cat", "p_dog"], "p_dog"),  # M5: 모르는 두 문자열이면 뒤쪽(dog)이 양성
        (["normal", "fraud", "normal", "fraud"], ["score_fraud", "score_normal"], "score_fraud"),
        (["no", "yes", "no", "yes"], ["Prob_No", "Prob_Yes"], "Prob_Yes"),
        ([0, 1, 0, 1], ["score_a", "score_b"], "score_a"),  # 이름이 안 맞으면 종전대로 첫 컬럼
    ],
)
def test_binary_two_score_columns_keep_the_positive_one(truth, columns, expected):
    """이진에서 양성 확률 후보가 둘이면 기본 양성 클래스 이름이 붙은 컬럼을 남긴다 — 첫 컬럼(prob_0)을 남기면
    음성 확률이 양성 확률로 들어가 AUROC 가 조용히 뒤집힌다(S9 검토)."""
    df = pd.DataFrame({"y_true": truth, **{c: [0.5] * len(truth) for c in columns}})
    resp = analyze_columns_fallback(task_type=B, columns=["y_true", *columns], df=df)
    roles = {m.column: m.role for m in resp.column_mappings}
    assert [c for c in columns if roles[c] == ColumnRole.score_positive] == [expected]
    other = next(c for c in columns if c != expected)
    assert roles[other] == ColumnRole.ignore
    notes = [n for n in resp.column_notes if n.llm_column == other]
    assert len(notes) == 1 and notes[0].status == "duplicate_role" and f"'{expected}'" in notes[0].message
    assert resp.metadata.positive_class is not None
