"""app/narrative/sections.py — 서술 소절 7칸 등록표(서술 설계 D36, 계획 설계 요지 1).

서술은 소절 7칸 단위로 만들고, 검사하고, 바꾸고, 기록한다. 이 표가 칸 ID(소절 번호)와 그 칸의 글이 응답의
어느 필드에 실리는지, 어느 장(章) 호출이 쓰는지, AI 응답의 어느 필드인지를 정한다. 필수 서술 항목 목록
(app.narrative.required_items)의 칸, 장별 지시문(app.narrative.prompt), 검문 사다리(app.narrative.generator),
규칙 템플릿(app.narrative.fallback.SECTION_TEMPLATES)이 이 표를 읽는다.

- 7.3(추가 관찰, D38)은 결론 목록 밖에서 AI 가 찾은 것만 쓰는 칸이다. AI 가 쓰지 않았거나 검문에 걸리면 성적서
  고정 문구 15("해당 없음")를 찍는다 — 규칙 문장이 없어 출처가 고정 문구다.
- 9칸은 9.1·9.2 서술과 권고표가 한 칸이다(사용자 결정 "D36 절 단위").

상호작용
- 의존(import): 없음(표준 라이브러리)
- 사용처: app.narrative.generator·prompt·fallback·jobs·required_items, tests
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    # 응답에서 이 칸의 글이 실리는 필드(점 표기). 9칸은 여럿이다.
    response_paths: tuple[str, ...]
    # 장(章) — 첫 생성은 장마다 한 번 부른다(7·8·9장, 계획 설계 요지 2).
    chapter: str
    # 작업 경로 AI 응답(app.narrative.prompt 의 칸 스키마)에서 이 칸의 필드 — response_paths 와 같은 차례.
    ai_fields: tuple[str, ...]
    # AI 가 쓰지 않으면(빈 글·검문 탈락·규칙 문장 경로) 고정 문구인 칸 — 7.3 은 규칙 문장이 없고 성적서가 고정 문구 15
    # ("해당 없음")를 찍는다(D38).
    fixed_without_ai: bool = False


SECTIONS: tuple[Section, ...] = (
    Section("7.1", "혼동행렬 해석", ("interpretation.confusion_analysis",), "7", ("section_7_1",)),
    Section("7.2", "분포·불균형", ("interpretation.distribution_analysis",), "7", ("section_7_2",)),
    Section("7.3", "추가 관찰", ("interpretation.additional_observations",), "7", ("section_7_3",), fixed_without_ai=True),
    Section("8.1", "벤치마크", ("conclusion.benchmark",), "8", ("section_8_1",)),
    Section("8.2", "종합 총평", ("conclusion.narrative",), "8", ("section_8_2",)),
    Section("8.3", "리스크", ("conclusion.risks",), "8", ("section_8_3",)),
    Section(
        "9",
        "권고",
        ("recommendation_narrative.data_quality", "recommendation_narrative.model_ops", "recommendations"),
        "9",
        ("section_9_1", "section_9_2", "recommendations"),
    ),
)

SECTION_IDS: tuple[str, ...] = tuple(s.id for s in SECTIONS)
SECTIONS_BY_ID: dict[str, Section] = {s.id: s for s in SECTIONS}
# 장 → 칸(첫 생성의 호출 단위). 문제가 0개면 9장은 부르지 않는다.
CHAPTERS: dict[str, tuple[str, ...]] = {
    ch: tuple(s.id for s in SECTIONS if s.chapter == ch) for ch in ("7", "8", "9")
}


# ── 판 4 자리(서술 설계 D47) ────────────────────────────────────────────────────────
# 칸 이름(위 id — 판 3 절 번호)은 판 4 에서도 그대로이고, 판 4 성적서에서 찍히는 자리만 바뀐다. 백엔드의 판 4 절 번호는
# 이 표 한 곳에서 읽는다(사실 보기의 '드러난 절'·'권고 절', 지시문·규칙 문장의 절 번호). 두 저장소 계약 사본
# tests/contract/narrative_slots_ed4.json 의 slots·recommend_sections 와 같다(시험이 대조).
EDITION4_PLACES: dict[str, str] = {
    "8.2": "5.1", "7.1": "5.2", "7.2": "5.3", "8.1": "5.4", "8.3": "5.5", "7.3": "5.6", "9": "6",
    "9.1": "6.1", "9.2": "6.2",
}


def place(section: str, *, edition: int) -> str:
    """칸 이름(판 3 절 번호)이 그 판의 성적서에서 찍히는 절 번호. 판 3 은 그대로, 판 4 는 EDITION4_PLACES(모르는 이름은 그대로)."""
    if edition == 4:
        return EDITION4_PLACES.get(section, section)
    return section


# 판 4 자리의 제목(예시본 5·6절 소제목 — 계약 narrative_slots_ed4.json 의 edition4_title 과 같다, 시험이 대조). 지시문이
# 칸을 부를 때('5.1 판정과 불확실성') 쓴다. 판 3 은 SECTIONS 의 title 그대로.
EDITION4_TITLES: dict[str, str] = {
    "8.2": "판정과 불확실성", "7.1": "오분류", "7.2": "데이터 분포", "8.1": "참조 범위", "8.3": "운영 위험",
    "7.3": "추가 관찰", "9": "개선 권고", "9.1": "데이터", "9.2": "모델과 운영",
}


def title(section: str, *, edition: int) -> str:
    """칸(또는 9.1·9.2)의 그 판 제목. 판 3 은 SECTIONS 의 title(9.1·9.2 는 판 3 칸이 아니라 KeyError)."""
    if edition == 4:
        return EDITION4_TITLES[section]
    return SECTIONS_BY_ID[section].title


# 판 4 성적서의 절 번호(목차 1~6절과 소절 — 계약 narrative_slots_ed4.json 의 edition4_section_numbers 와 같다, 시험이 대조).
# 숫자 검문이 판 4 글에서 이 번호를 숫자로 보지 않는다(app.narrative.grounding — 계획 2.7).
EDITION4_SECTION_NUMBERS: tuple[str, ...] = (
    "1", "1.1", "1.2", "1.3",
    "2", "2.1", "2.2", "2.3",
    "3", "3.1", "3.2", "3.3", "3.4", "3.5",
    "4", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8",
    "5", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6",
    "6", "6.1", "6.2",
)
