"""
analyzer.py — 파일 파싱 + 메타데이터 추출 + LLM 호출 핵심 로직
"""

import json
import io
import re
import pandas as pd
from openai import AsyncOpenAI

from app.core.schemas import (
    AnalysisResponse, ColumnMapping, ColumnMatchNote, ColumnRole,
    DataMetadata, TaskType, VALID_ROLES_BY_TASK,
)
from app.analysis.prompt_builder import build_system_prompt, build_user_prompt


def _build_response_schema(task_type: TaskType, columns: list[str] | None = None) -> dict:
    """task_type에 맞는 role만 허용하는 JSON Schema 생성.

    columns 를 주면 column 값을 실제 헤더 enum 으로 제약해 LLM 환각 컬럼명을 원천 차단한다(D5a).
    (동적 enum strict 가 거부되면 호출부가 columns=None 으로 1회 재시도한다.)
    """
    valid_roles = [r.value for r in VALID_ROLES_BY_TASK[task_type]]
    column_prop: dict = {"type": "string"}
    if columns:
        column_prop = {"type": "string", "enum": columns}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "column_mapping_result",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "column_mappings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": column_prop,
                                "role":   {"type": "string", "enum": valid_roles},
                            },
                            "required": ["column", "role"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["column_mappings"],
                "additionalProperties": False,
            },
        },
    }


def _norm(s: str) -> str:
    """컬럼명 정규화: 공백/밑줄/하이픈 제거 + 소문자. (대소문자·구분자 변형 매칭용)"""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def _reconcile_llm_columns(
    llm_mappings: list[dict], actual_cols: list[str]
) -> tuple[list[dict], list[ColumnMatchNote]]:
    """LLM 반환 컬럼명을 실제 헤더에 정렬한다(신뢰 경계 검증, D5a).

    - 정확 일치 → 그대로. 정규화(대소문자/공백/_/-) 일치 → 실제 헤더로 보정(corrected).
    - 실제 헤더에 없으면 매핑에서 제외(unmatched) — 환각 컬럼명이 결과에 들어가지 않게.
    - LLM 이 한 번도 반환하지 않은 실제 헤더는 ignore 로 보완(unmapped_header) — 조용한 컬럼 소실 방지.
    (difflib 유사 매칭은 무음 오매핑 위험이 있어 advisory 로 분리, 여기서는 자동 치환하지 않음.)
    """
    notes: list[ColumnMatchNote] = []
    actual_set = set(actual_cols)

    # 정규화 사전(충돌 키는 모호하므로 정규화 매칭 대상에서 제외)
    norm_to_actual: dict[str, str] = {}
    collisions: set[str] = set()
    for c in actual_cols:
        n = _norm(c)
        if n in norm_to_actual:
            collisions.add(n)
        else:
            norm_to_actual[n] = c
    for n in collisions:
        norm_to_actual.pop(n, None)

    used: set[str] = set()
    reconciled: list[dict] = []
    for m in llm_mappings:
        col = m.get("column", "")
        role = m.get("role", ColumnRole.ignore.value)
        if col in actual_set and col not in used:
            reconciled.append({"column": col, "role": role})
            used.add(col)
            continue
        matched = norm_to_actual.get(_norm(col))
        if matched and matched not in used:
            reconciled.append({"column": matched, "role": role})
            used.add(matched)
            notes.append(ColumnMatchNote(
                llm_column=col, matched_column=matched, status="corrected",
                message=f"'{col}'을(를) 실제 컬럼 '{matched}'(으)로 보정했습니다.",
            ))
            continue
        notes.append(ColumnMatchNote(
            llm_column=col, matched_column=None, status="unmatched",
            message=f"'{col}'은(는) 파일 헤더에 없어 매핑에서 제외했습니다. 필요 시 직접 지정하세요.",
        ))

    # 미반환 실제 헤더 → ignore 로 보완(UI 에서 역할 지정 가능)
    for c in actual_cols:
        if c not in used:
            reconciled.append({"column": c, "role": ColumnRole.ignore.value})
            notes.append(ColumnMatchNote(
                llm_column="", matched_column=c, status="unmapped_header",
                message=f"'{c}'은(는) 자동 매핑되지 않아 '무시(ignore)'로 추가했습니다. 필요 시 역할을 지정하세요.",
            ))

    return reconciled, notes


def parse_file_content(file_content: bytes, filename: str) -> tuple[list[str], pd.DataFrame]:
    """
    CSV 또는 JSON 파일을 파싱해 컬럼명 목록과 전체 DataFrame을 반환합니다.

    지원 JSON 구조:
      1. records 배열:  [{col: val, ...}, ...]
      2. 열 기반 dict:  {col: [val, ...], ...}
      3. 단일 키 래핑:  {"samples": [{...}, ...]}  ← 자동 언래핑
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        # 인코딩 자동 감지: UTF-8 → CP949(한국어 Excel) → latin-1 순으로 시도
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue
        else:
            raise ValueError(f"파일 인코딩을 자동으로 감지할 수 없습니다: {last_error}")

    elif ext == "json":
        raw = json.loads(file_content.decode("utf-8"))

        if isinstance(raw, dict):
            values = list(raw.values())
            if len(values) == 1 and isinstance(values[0], list):
                raw = values[0]

        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            df = pd.DataFrame(raw)
        else:
            raise ValueError(
                "JSON 형식 오류: records 배열([{...}]) 또는 열 기반 dict({col: [...]}) 형태여야 합니다."
            )
    else:
        raise ValueError(f"지원하지 않는 파일 형식: .{ext}  (CSV 또는 JSON만 허용)")

    # 전체 DataFrame 반환 (메타데이터 추출용)
    return df.columns.tolist(), df


# ── 양성 클래스 자동 판단 규칙 ────────────────────────────────────────────────
# 숫자형: 큰 값이 Positive (1 > 0, True > False)
# 문자열 known patterns: 아래에 정의된 것만 자동 판단, 나머지는 ambiguous 처리
_KNOWN_POSITIVE = {"1", "yes", "true", "positive", "pos", "spam", "malignant", "fraud", "1.0"}
_KNOWN_NEGATIVE = {"0", "no", "false", "negative", "neg", "ham", "benign", "normal", "0.0"}


def _detect_binary_classes(series: pd.Series) -> tuple[str | None, str | None, bool]:
    """
    Binary y_true 컬럼에서 양성/음성 클래스를 자동 추론합니다.

    Returns:
        (positive_class, negative_class, is_ambiguous)
        is_ambiguous=True 이면 사용자 확인이 필요합니다.
    """
    unique_vals = [str(v) for v in series.dropna().unique()]
    if len(unique_vals) != 2:
        return None, None, True  # 2개 값이 아니면 판단 불가

    a, b = unique_vals[0], unique_vals[1]
    a_lower, b_lower = a.lower(), b.lower()

    # 숫자형: 큰 값이 Positive
    try:
        fa, fb = float(a), float(b)
        if fa > fb:
            return a, b, False
        else:
            return b, a, False
    except ValueError:
        pass

    # Known pattern 매칭
    if a_lower in _KNOWN_POSITIVE and b_lower in _KNOWN_NEGATIVE:
        return a, b, False
    if b_lower in _KNOWN_POSITIVE and a_lower in _KNOWN_NEGATIVE:
        return b, a, False

    # 판단 불가 → 알파벳순으로 첫 번째를 Positive로 임시 지정 후 ambiguous 표시
    pos, neg = sorted([a, b])
    return pos, neg, True


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

    # [설계 개선] 파일 내 모든 컬럼에 대해 전체 유니크값 목록을 미리 계산해 둡니다.
    # 사용자가 화면에서 컬럼 매핑을 변경(ignore -> y_true)하더라도 누락 없이 전체 클래스 목록을 볼 수 있게 지원합니다.
    column_unique_values: dict[str, list[str]] = {}
    for col in df.columns:
        non_null_series = df[col].dropna()
        if non_null_series.empty:
            column_unique_values[col] = []
            continue
        unique_set = set()
        for val in non_null_series:
            val_str = str(val).strip()
            if not val_str:
                continue
            if task_type == TaskType.multilabel:
                # 멀티레이블은 파이프로 쪼개서 원소 수집
                for part in val_str.split('|'):
                    part = part.strip()
                    if part:
                        unique_set.add(part)
            else:
                unique_set.add(val_str)
        column_unique_values[col] = sorted(list(unique_set))

    # ── Binary ────────────────────────────────────────────────────────────────
    if task_type == TaskType.binary:
        y_true_col = role_to_col.get(ColumnRole.y_true.value)
        if y_true_col and y_true_col in df.columns:
            # 클래스 감지: 샘플 30행으로
            pos, neg, ambiguous = _detect_binary_classes(sample_df[y_true_col])
            # 분포: 전체 df로
            distribution = df[y_true_col].value_counts().to_dict()
            distribution = {str(k): int(v) for k, v in distribution.items()}
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
            distribution = df[y_true_col].value_counts().to_dict()
            distribution = {str(k): int(v) for k, v in distribution.items()}
            # 전체 분포의 키값들을 기반으로 클래스 목록 감지 (30행 제한 제거)
            classes = sorted(distribution.keys())
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
                for label in str(cell).split("|"):
                    label = label.strip()
                    if label:
                        label_counts[label] = label_counts.get(label, 0) + 1
            labels = sorted(label_counts.keys())
            return DataMetadata(
                detected_labels=labels,
                class_distribution=label_counts,
                column_unique_values=column_unique_values,
            )

    return DataMetadata(column_unique_values=column_unique_values)


async def analyze_columns_with_llm(
    client: AsyncOpenAI,
    task_type: TaskType,
    columns: list[str],
    df: pd.DataFrame,
) -> AnalysisResponse:
    """LLM으로 컬럼 역할을 자동 매핑하고, 데이터 메타데이터를 추출합니다."""
    # 30행 샘플: LLM 컬럼 매핑 + 클래스 감지용
    sample_df = df.head(30)

    messages = [
        {"role": "system", "content": build_system_prompt(task_type)},
        {"role": "user",   "content": build_user_prompt(columns, sample_df)},
    ]
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            response_format=_build_response_schema(task_type, columns),
            temperature=0,
        )
    except Exception:
        # 동적 enum strict 스키마가 거부되면 enum 없는 스키마로 1회 재시도(D5a).
        # (그래도 실패하면 라우터의 규칙 폴백으로 강등된다.)
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            response_format=_build_response_schema(task_type, None),
            temperature=0,
        )

    data = json.loads(response.choices[0].message.content)

    # LLM 반환 컬럼명을 실제 헤더에 정렬(환각/변형 컬럼명 차단, 미반환 헤더 보완) — D5a
    reconciled, notes = _reconcile_llm_columns(data["column_mappings"], columns)
    column_mappings = []
    for m in reconciled:
        col_name = m["column"]
        samples = []
        if col_name in df.columns:
            # 결측치를 제거하고 상위 3개 값을 문자열로 변환하여 예시 데이터 추출
            samples = [str(v) for v in df[col_name].dropna().head(3).tolist()]
        column_mappings.append(
            ColumnMapping(
                column=col_name,
                role=ColumnRole(m["role"]),
                sample_values=samples
            )
        )

    # 클래스 감지: sample_df(30행) / 분포 계산: 전체 df
    metadata = extract_metadata(task_type, df, sample_df, column_mappings)

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata,
        column_notes=notes,
    )


def analyze_columns_fallback(
    task_type: TaskType,
    columns: list[str],
    df: pd.DataFrame,
) -> AnalysisResponse:
    """OpenAI API 키가 없을 때 작동하는 규칙 기반 컬럼 매핑 폴백 함수"""
    column_mappings = []

    for col in columns:
        col_lower = col.lower()
        role = ColumnRole.ignore

        if "id" in col_lower or "index" in col_lower:
            role = ColumnRole.sample_id
        elif col_lower in ["y_true", "actual", "ground_truth", "label", "target"]:
            if task_type == TaskType.multilabel:
                role = ColumnRole.true_labels
            else:
                role = ColumnRole.y_true
        elif col_lower in ["y_pred", "predicted", "pred", "prediction"]:
            if task_type == TaskType.multilabel:
                role = ColumnRole.pred_labels
            else:
                role = ColumnRole.y_pred
        elif task_type == TaskType.binary and ("score" in col_lower or "prob" in col_lower or "pos" in col_lower):
            role = ColumnRole.score_positive
        elif task_type == TaskType.multiclass and ("prob" in col_lower or "p_" in col_lower or "class_" in col_lower):
            role = ColumnRole.prob_per_class
        elif task_type == TaskType.multilabel and ("score" in col_lower or "prob" in col_lower or "p_" in col_lower):
            role = ColumnRole.score_per_label

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

    return AnalysisResponse(
        task_type=task_type,
        column_mappings=column_mappings,
        metadata=metadata
    )
