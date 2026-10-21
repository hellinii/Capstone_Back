"""app/narrative/sections.py — 서술 소절 7칸 등록표(서술 설계 D36, 계획 설계 요지 1).

서술은 소절 7칸 단위로 만들고, 검사하고, 바꾸고, 기록한다. 이 표가 칸 ID(소절 번호)와 그 칸의 글이 응답의
어느 필드에 실리는지를 정한다. 필수 서술 항목 목록(app.narrative.required_items)의 칸, 작업 경로의 칸별
출처(app.narrative.jobs.section_sources)가 이 표를 읽는다. S5 에서 칸마다 검사 종류를, S6 에서 템플릿 함수를
채운다.

- 7.3(추가 관찰, D38)은 결론 목록 밖에서 AI 가 찾은 것만 쓰는 칸이다. AI 가 쓰지 않았거나 검문에 걸리면 성적서
  고정 문구 15("해당 없음")를 찍는다 — 그래서 AI 가 7.3 을 쓰기 전(S5 이전)에는 출처가 늘 고정 문구다.
- 9칸은 9.1·9.2 서술과 권고표가 한 칸이다(사용자 결정 "D36 절 단위").

상호작용
- 의존(import): 없음(표준 라이브러리)
- 사용처: app.narrative.jobs(SECTION_IDS·칸별 출처), app.narrative.required_items(칸 이름), tests
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    # 응답에서 이 칸의 글이 실리는 필드(점 표기). 9칸은 여럿이다.
    response_paths: tuple[str, ...]
    # AI 가 이 칸을 쓰기 전까지 늘 고정 문구인 칸(7.3 — S5 에서 AI 가 쓰면 False 로 바뀐다).
    fixed_until_generated: bool = False


SECTIONS: tuple[Section, ...] = (
    Section("7.1", "혼동행렬 해석", ("interpretation.confusion_analysis",)),
    Section("7.2", "분포·불균형", ("interpretation.distribution_analysis",)),
    Section("7.3", "추가 관찰", ("interpretation.additional_observations",), fixed_until_generated=True),
    Section("8.1", "벤치마크", ("conclusion.benchmark",)),
    Section("8.2", "종합 총평", ("conclusion.narrative",)),
    Section("8.3", "리스크", ("conclusion.risks",)),
    Section(
        "9",
        "권고",
        ("recommendation_narrative.data_quality", "recommendation_narrative.model_ops", "recommendations"),
    ),
)

SECTION_IDS: tuple[str, ...] = tuple(s.id for s in SECTIONS)
SECTIONS_BY_ID: dict[str, Section] = {s.id: s for s in SECTIONS}
