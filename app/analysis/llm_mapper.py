"""app/analysis/llm_mapper.py — LLM 기반 컬럼 자동 매핑(외부 경계)

OpenAI 호출로 컬럼 역할을 매핑하고(거부 시 enum 없는 스키마로 1회 재시도), 반환 컬럼명을
reconcile 로 실제 헤더에 정렬한 뒤 메타데이터를 붙여 AnalysisResponse 를 만든다.

상호작용
- 의존(import): openai, pandas, app.core.schemas, app.analysis.prompt_builder,
  app.analysis.reconcile(reconcile_llm_columns), app.analysis.metadata(extract_metadata)
- 사용처: app.analysis.analysis_service / scripts.llm_smoke_analyze
"""
import json

import openai
import pandas as pd
from openai import AsyncOpenAI

from app.core.schemas import ColumnMapping, ColumnRole, TaskType, VALID_ROLES_BY_TASK
from app.analysis.schemas import AnalysisResponse
from app.analysis.prompt_builder import build_system_prompt, build_user_prompt, MAX_PROMPT_COLUMNS
from app.core.concurrency import llm_slot
from app.analysis.reconcile import reconcile_llm_columns
from app.analysis.metadata import extract_metadata


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
    # 컬럼이 지나치게 많으면 동적 enum 자체가 증폭 수단이 된다 — enum 을 포기하고
    # reconcile 의 컬럼명 정렬에 맡긴다(G-03).
    schema_columns = columns if len(columns) <= MAX_PROMPT_COLUMNS else None

    try:
        async with llm_slot():
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                response_format=_build_response_schema(task_type, schema_columns),
                temperature=0,
            )
    except (openai.BadRequestError, openai.UnprocessableEntityError):
        # 동적 enum strict 스키마가 **거부된 경우에만** enum 없는 스키마로 1회 재시도(D5a).
        # 종전에는 blanket `except Exception` 이라 타임아웃·레이트리밋까지 재시도해
        # SDK max_retries=2 와 곱해지면서 익명 요청 1건의 과금 호출이 최대 6회,
        # 최악 벽시계가 약 270초였다(E-18). 그런 실패는 그대로 위로 던져
        # analysis_service 의 규칙 폴백으로 즉시 강등시킨다.
        async with llm_slot():
            response = await client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                response_format=_build_response_schema(task_type, None),
                temperature=0,
            )

    data = json.loads(response.choices[0].message.content)

    # LLM 반환 컬럼명을 실제 헤더에 정렬(환각/변형 컬럼명 차단, 미반환 헤더 보완) — D5a
    reconciled, notes = reconcile_llm_columns(data["column_mappings"], columns)
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
