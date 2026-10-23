"""app/analysis/fallback_mapper.py — 규칙 기반 컬럼 매핑 폴백

OpenAI 키가 없거나 LLM 호출이 실패했을 때 컬럼명 패턴으로 역할을 추정하는 폴백 매핑.

상호작용
- 의존(import): pandas, app.core.schemas, app.analysis.metadata(extract_metadata),
  app.evaluation.labels(default_positive_class), app.evaluation.preprocessor(PROB_COLUMN_PREFIXES)
- 사용처: app.analysis.analysis_service (무키/LLM 실패 시 강등 경로)
"""
import re

import pandas as pd

from app.core.schemas import ColumnMapping, ColumnRole, MULTI_COLUMN_ROLES, TaskType
from app.analysis.schemas import AnalysisResponse, ColumnMatchNote
from app.analysis.metadata import extract_metadata
from app.evaluation.labels import default_positive_class, normalize_label
from app.evaluation.preprocessor import PROB_COLUMN_PREFIXES


# ── 이름 규칙(M8·M4) ──────────────────────────────────────────────────────────
# 종전에는 부분 문자열로 봐서 이름에 'id' 가 든 컬럼(valid_flag·width·hidden …)을 모두 샘플 ID 로,
# 'pos' 가 든 컬럼(position …)을 양성 확률로 잡았다. 이제 이름을 **낱말**로 쪼개서 본다 —
# 구분자(_ - 공백 .)와 camelCase 경계(imageId → image, id).

# 낱말 하나라도 이것이면 샘플 ID — id·image_id·imageId·row_index·uid.
SAMPLE_ID_WORDS = frozenset({"id", "idx", "index", "uid", "uuid"})
# 구분자 없이 붙여 쓴 식별자 이름(낱말 하나) — 종전 부분 문자열 규칙이 잡던 흔한 꼴을 놓치지 않게 따로 둔다.
# 'id' 로 끝나는 낱말을 모두 잡으면 valid·grid·void 를 다시 오인하므로 목록으로만 받는다(S9 검토).
SAMPLE_ID_COMPOUNDS = frozenset({
    "sampleid", "rowid", "rowidx", "rowindex", "imageid", "imgid", "fileid", "userid",
    "itemid", "recordid", "docid", "caseid", "objectid", "instanceid",
})

# 이름 전체(낱말을 _ 로 이은 꼴)가 이것이면 정답·예측. 멀티레이블은 true_labels·pred_labels 역할로 간다.
TRUTH_NAMES = frozenset({
    "y_true", "actual", "ground_truth", "label", "target",
    "true_label", "true_labels", "y_true_labels", "gold_labels",
})
PREDICTION_NAMES = frozenset({
    "y_pred", "predicted", "pred", "prediction",
    "pred_label", "pred_labels", "predicted_labels", "y_pred_labels",
})

# 지연시간(M4). 단위는 ms 로 가정한다(ColumnRole.latency) — 이름으로 단위를 확정할 수는 없어서
# 값이 모두 1 미만이면 검증 단계가 "초 단위로 보임" 경고를 낸다(validation_checks).
LATENCY_NAMES = frozenset({
    "latency", "latency_ms", "inference_time", "inference_time_ms",
    "response_time", "response_time_ms", "elapsed_ms",
})

# 확률·점수 낱말. 'pos'·'p' 처럼 짧은 낱말은 다른 뜻(position 의 줄임 등)과 겹쳐 넣지 않는다.
PROBABILITY_WORDS = frozenset({"prob", "proba", "probability", "score"})
# 복수형·숫자 꼬리(probs·scores·prob1)도 같은 낱말로 본다 — 종전 부분 문자열 규칙이 잡던 꼴(S9 검토).
_PROBABILITY_WORD_RE = re.compile(r"^(probs?|probas?|probabilit(y|ies)|scores?)\d*$")


def _is_probability_word(word: str) -> bool:
    return word in PROBABILITY_WORDS or bool(_PROBABILITY_WORD_RE.match(word))


def name_words(column: str) -> list[str]:
    """컬럼 이름을 소문자 낱말로 쪼갠다 — 'imageId' → ['image', 'id'], 'row-index' → ['row', 'index']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column.strip())
    return [w for w in re.split(r"[_\-\s.]+", spaced.lower()) if w]


def fallback_role(task_type: TaskType, column: str) -> ColumnRole:
    """컬럼 이름 하나만 보고 역할을 고른다(규칙 폴백). 모르면 ignore."""
    words = name_words(column)
    joined = "_".join(words)
    multilabel = task_type == TaskType.multilabel

    if joined in TRUTH_NAMES:
        return ColumnRole.true_labels if multilabel else ColumnRole.y_true
    if joined in PREDICTION_NAMES:
        return ColumnRole.pred_labels if multilabel else ColumnRole.y_pred
    if joined in LATENCY_NAMES or "latency" in words:
        return ColumnRole.latency
    if SAMPLE_ID_WORDS.intersection(words) or joined in SAMPLE_ID_COMPOUNDS:
        return ColumnRole.sample_id
    # 'p_cat' 처럼 p 하나로 시작하는 꼴은 평가가 받는 확률 접두어(p_)다.
    if any(_is_probability_word(w) for w in words) or (len(words) > 1 and words[0] == "p"):
        return {
            TaskType.binary: ColumnRole.score_positive,
            TaskType.multiclass: ColumnRole.prob_per_class,
            TaskType.multilabel: ColumnRole.score_per_label,
        }[task_type]
    if task_type == TaskType.binary and "positive" in words:
        return ColumnRole.score_positive
    return ColumnRole.ignore


def _preferred_score_column(
    task_type: TaskType, roles: dict[str, ColumnRole], df: pd.DataFrame,
) -> str | None:
    """이진에서 양성 확률 후보가 여럿(prob_0·prob_1)이면 기본 양성 클래스 이름이 붙은 컬럼을 고른다.

    '첫 컬럼만 남김' 규칙을 그대로 쓰면 prob_0·prob_1 에서 음성 확률(prob_0)이 양성 확률로 들어가
    AUROC 등이 조용히 뒤집힌다(S9 검토). 양성은 자동 판단·지표 계산과 같은 함수(default_positive_class)로
    정하고, 이름이 맞는 후보가 없으면 None(종전대로 첫 컬럼).
    """
    if task_type != TaskType.binary:
        return None
    candidates = [c for c, r in roles.items() if r == ColumnRole.score_positive]
    truth = next((c for c, r in roles.items() if r == ColumnRole.y_true), None)
    if len(candidates) < 2 or truth is None or truth not in df.columns:
        return None
    positive, negative, _ = default_positive_class(df[truth].dropna().unique())
    if positive is None or negative is None:
        return None
    target = normalize_label(positive).lower()
    for col in candidates:
        lowered = col.lower()
        for prefix in PROB_COLUMN_PREFIXES:
            if lowered.startswith(prefix) and lowered[len(prefix):] == target:
                return col
    return None


def analyze_columns_fallback(
    task_type: TaskType,
    columns: list[str],
    df: pd.DataFrame,
) -> AnalysisResponse:
    """OpenAI API 키가 없을 때 작동하는 규칙 기반 컬럼 매핑 폴백 함수"""
    column_mappings = []
    # 한 컬럼만 받는 역할을 두 번째 컬럼이 또 잡으면 매핑 확정이 막힌다(역할 중복). 첫 컬럼만 그 역할로
    # 두고 나머지는 ignore 로 내린 뒤 알린다(M8) — 사용자는 매핑 화면에서 바꿀 수 있다.
    duplicate_notes: list[ColumnMatchNote] = []
    roles = {col: fallback_role(task_type, col) for col in columns}
    taken: dict[ColumnRole, str] = {}
    preferred_score = _preferred_score_column(task_type, roles, df)
    if preferred_score is not None:
        taken[ColumnRole.score_positive] = preferred_score

    for col in columns:
        role = roles[col]
        if role not in MULTI_COLUMN_ROLES:
            if taken.get(role) == col:
                pass  # 미리 고른 컬럼(이진 양성 확률) — 그대로 둔다
            elif role in taken:
                duplicate_notes.append(
                    ColumnMatchNote(
                        llm_column=col,
                        matched_column=None,
                        status="duplicate_role",
                        message=(
                            f"'{col}' 컬럼도 '{role.value}' 역할로 보였지만 그 역할은 '{taken[role]}' 컬럼이 "
                            "먼저 맡아 평가에서 제외했습니다(자동 매핑이 규칙 기반으로 동작했습니다). "
                            "필요하면 매핑 화면에서 직접 지정해 주세요."
                        ),
                    )
                )
                role = ColumnRole.ignore
            else:
                taken[role] = col

        samples = []
        if col in df.columns:
            samples = [str(v) for v in df[col].dropna().head(3).tolist()]

        column_mappings.append(
            ColumnMapping(
                column=col,
                role=role,
                sample_values=samples
            )
        )

    sample_df = df.head(30)
    metadata = extract_metadata(task_type, df, sample_df, column_mappings)

    # 어느 규칙에도 걸리지 않은 컬럼은 조용히 ignore 가 된다 — 그 사실을 알린다
    # (B-03). elif 사슬에 else 가 없어 종전에는 안내가 **항상 비어 있었고**,
    # 무키·예산 소진처럼 **가장 안내가 필요한 순간에** 사용자는 아무 설명도 못 받았다.
    # (LLM 경로의 reconcile 은 오히려 성실히 안내를 만든다 — 비어 있던 것은 이쪽이다.)
    column_notes = [
        ColumnMatchNote(
            llm_column=m.column,
            matched_column=None,
            status="unmapped_header",
            message=(
                f"'{m.column}' 컬럼의 역할을 규칙으로 판단하지 못해 평가에서 제외했습니다"
                "(자동 매핑이 규칙 기반으로 동작했습니다). 필요하면 매핑 화면에서 직접 지정해 주세요."
            ),
        )
        for m in column_mappings
        if m.role == ColumnRole.ignore and m.column not in {n.llm_column for n in duplicate_notes}
    ] + duplicate_notes

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata,
        column_notes=column_notes,
    )
