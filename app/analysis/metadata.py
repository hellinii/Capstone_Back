"""app/analysis/metadata.py — 확정 매핑 기반 데이터 메타데이터 추출

업로드 데이터에서 task_type 별 클래스/레이블/분포와 컬럼별 유니크값을 계산한다(순수 로직).
binary 는 양성/음성 클래스 자동 판단(_detect_binary_classes)까지 수행한다.

상호작용
- 의존(import): pandas, app.core.schemas(ColumnMapping, ColumnRole, DataMetadata, TaskType)
- 사용처: app.analysis.llm_mapper / app.analysis.fallback_mapper (매핑 후 메타 추출)
"""
import pandas as pd

from app.core.schemas import ColumnMapping, ColumnRole, DataMetadata, TaskType
from app.evaluation.labels import (
    default_positive_class, normalize_distribution, normalize_label, parse_label_cell, sort_labels,
)


# 컬럼별 고유값 목록을 담는 상한. 이 목록은 "이 컬럼을 라벨로 쓰면 어떤 클래스가 있는가"를
# 보여주기 위한 것이라, 이보다 값이 다양한 컬럼(id, score 등)은 라벨 후보가 아니다.
MAX_UNIQUE_VALUES_PER_COLUMN = 200


def _detect_binary_classes(series: pd.Series) -> tuple[str | None, str | None, bool]:
    """
    Binary y_true 컬럼에서 양성/음성 클래스를 자동 추론합니다 — 규칙은 labels.default_positive_class
    하나다(M5: 지표 계산·예측 파생과 같은 값). 값이 둘이 아니면 판단하지 않는다(None, None, True).

    Returns:
        (positive_class, negative_class, is_ambiguous)
        is_ambiguous=True 이면 사용자 확인이 필요합니다.
    """
    pos, neg, ambiguous = default_positive_class(series.dropna().unique())
    if neg is None:
        return None, None, True  # 2개 값이 아니면 판단 불가
    # 표기는 column_unique_values 와 같은 normalize_label — str(1.0) = '1.0' 이면 프론트가 목록에 없는 양성으로
    # 보고 비운다(빈 칸 섞인 정수 정답). 평가 쪽은 labels.find_label 로 원래 값을 되찾는다.
    return normalize_label(pos), normalize_label(neg), ambiguous


def extract_metadata(
    task_type: TaskType,
    df: pd.DataFrame,
    sample_df: pd.DataFrame,
    column_mappings: list[ColumnMapping],
) -> DataMetadata:
    """
    확정된 컬럼 매핑을 기반으로 메타데이터를 추출합니다.

    - 클래스 감지: sample_df (30행) 기준 → 속도 우선
    - 분포 계산:   df (전체)     기준 → 정확도 우선

    - Binary:     양성/음성 클래스 자동 판단
    - Multiclass: y_true 고유 클래스 목록 + 분포
    - Multilabel: true_labels 파싱 후 고유 레이블 목록 + 분포
    """
    role_to_col: dict[str, str] = {m.role.value: m.column for m in column_mappings}

    # [설계 개선] 파일 내 모든 컬럼에 대해 유니크값 목록을 미리 계산해 둡니다.
    # 사용자가 화면에서 컬럼 매핑을 변경(ignore -> y_true)하더라도 누락 없이 전체 클래스 목록을 볼 수 있게 지원합니다.
    #
    # 고유값이 상한을 넘는 컬럼은 담지 않습니다. 이 목록의 용도는 "이 컬럼을 라벨로 쓰면
    # 어떤 클래스가 있는가"를 보여주는 것이라, id·score 처럼 값이 행마다 다른 컬럼은 애초에
    # 후보가 아닙니다. 상한 없이 담으면 id 컬럼 하나가 행 수만큼 들어가 metadata 가 수백 KB로
    # 불어나고, 이 값이 평가 run 마다 브라우저 localStorage 에 저장되어 용량 한도를 넘기면
    # 성적서 생성이 실패합니다. 담지 않은 컬럼은 프론트가 detected_classes/샘플값으로 폴백합니다.
    column_unique_values: dict[str, list[str]] = {}
    for col in df.columns:
        non_null_series = df[col].dropna()
        if non_null_series.empty:
            column_unique_values[col] = []
            continue
        unique_set: set[str] = set()
        overflowed = False
        for val in non_null_series:
            val_str = str(val).strip()
            if not val_str:
                continue
            if task_type == TaskType.multilabel:
                # 파싱 규칙은 evaluation.labels 하나뿐이다(ISSUES.md D-04). 종전에는 여기가
                # '|' 만 쪼개서, 평가 파서가 라벨 2개로 읽는 "a,b" 를 라벨 1개로 셌다.
                unique_set.update(parse_label_cell(val))
            else:
                unique_set.add(normalize_label(val))
            if len(unique_set) > MAX_UNIQUE_VALUES_PER_COLUMN:
                overflowed = True
                break
        if overflowed:
            continue  # 라벨 후보가 아니므로 목록에서 제외
        column_unique_values[col] = sort_labels(unique_set)

    # ── Binary ────────────────────────────────────────────────────────────────
    if task_type == TaskType.binary:
        y_true_col = role_to_col.get(ColumnRole.y_true.value)
        if y_true_col and y_true_col in df.columns:
            # 클래스 감지: 샘플 30행으로
            pos, neg, ambiguous = _detect_binary_classes(sample_df[y_true_col])
            # 분포: 전체 df로
            distribution = normalize_distribution(df[y_true_col].value_counts().to_dict())
            return DataMetadata(
                positive_class=pos,
                negative_class=neg,
                positive_class_ambiguous=ambiguous,
                class_distribution=distribution,
                column_unique_values=column_unique_values,
            )

    # ── Multiclass ────────────────────────────────────────────────────────────
    elif task_type == TaskType.multiclass:
        y_true_col = role_to_col.get(ColumnRole.y_true.value)
        if y_true_col and y_true_col in df.columns:
            # 분포: 전체 df로
            distribution = normalize_distribution(df[y_true_col].value_counts().to_dict())
            # 전체 분포의 키값들을 기반으로 클래스 목록 감지 (30행 제한 제거).
            # 정렬은 sort_labels 가 정본 — 숫자 클래스가 ['1','10','2'] 로 뒤집히지 않는다.
            classes = sort_labels(distribution.keys())
            return DataMetadata(
                detected_classes=classes,
                class_distribution=distribution,
                column_unique_values=column_unique_values,
            )

    # ── Multilabel ────────────────────────────────────────────────────────────
    elif task_type == TaskType.multilabel:
        true_col = role_to_col.get(ColumnRole.true_labels.value)
        if true_col and true_col in df.columns:
            # 분포: 전체 df로 계산하면서 동시에 전체 라벨 감지 (30행 제한 제거)
            label_counts: dict[str, int] = {}
            for cell in df[true_col].dropna():
                for label in parse_label_cell(cell):
                    label_counts[label] = label_counts.get(label, 0) + 1
            label_counts = normalize_distribution(label_counts)
            labels = sort_labels(label_counts.keys())
            return DataMetadata(
                detected_labels=labels,
                class_distribution=label_counts,
                column_unique_values=column_unique_values,
            )

    return DataMetadata(column_unique_values=column_unique_values)
