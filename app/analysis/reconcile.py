"""app/analysis/reconcile.py — LLM 반환 컬럼명을 실제 파일 헤더에 정렬(신뢰 경계)

LLM이 돌려준 컬럼 매핑의 컬럼명이 실제 업로드 파일 헤더와 어긋날 수 있으므로(대소문자·
공백·구분자 변형, 환각 컬럼명 등), 이를 실제 헤더에 맞춰 보정·정렬하는 순수 로직 모음이다.
LLM/IO 의존이 없어 단독 단위 테스트가 가능하다.

상호작용
- 의존(import): app.core.schemas(ColumnRole, ColumnMatchNote)
- 사용처: app.analysis.analyzer(analyze_columns_with_llm 이 LLM 응답 보정에 사용), tests
"""
import re

from app.core.schemas import ColumnMatchNote, ColumnRole


def _norm(s: str) -> str:
    """컬럼명 정규화: 공백/밑줄/하이픈 제거 + 소문자. (대소문자·구분자 변형 매칭용)"""
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


def reconcile_llm_columns(
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
