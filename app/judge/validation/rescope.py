"""app/judge/validation/rescope.py — 재검증 범위 계산(계획 정10, 설계 7.7 "다시 도는 범위"). 순수 함수.

규칙(정10): 바뀐 모듈과 그 출력을 받는 모듈을 모든 실행(재현성·결함·sham)에서 다시 돈다. 주장 추출이나 공통 설정(모델·추론 강도)을
고치면 4종 전부(추출 재현율·정밀도까지). 바뀐 축의 LLM 판정이 달라졌으면 그 축의 사람 일치도 표본도 다시 채점한다.

채점 한 번(judge_once)은 늘 네 축을 함께 돌리므로 "다시 돌 모듈"은 다시 비교할 결과의 범위이고, 다시 부르는 채점 횟수는 묶음
회차 그대로다. 표에 없는 심판 코드(app/judge 아래 새 파일)는 모르는 것이라 공통으로 본다(덜 돌리는 쪽으로 틀리지 않게).

상호작용
- 의존(import): fnmatch
- 사용처: scripts/judge_validate.py rescope, 시험 tests/test_judge_validation_rescope.py
"""
from __future__ import annotations

from fnmatch import fnmatch

AXES = ("faithfulness", "consistency", "coverage", "readability")
ALL_BATCHES = ("reproducibility", "defects", "sham", "extraction")
RUN_BATCHES = ("reproducibility", "defects", "sham")
C = "app/judge/criteria/"

# (경로 꼴, 바뀌는 모듈). 위에서부터 처음 맞는 줄. 모듈: common·extract·네 축·aggregate(합산·게이트·안전장치). 빈 튜플 = 영향 없음.
TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 기준 묶음 밖 기록(지문에 들지 않음)·버전 이름만
    (C + "CHANGELOG.md", ()), (C + "VERSIONS.json", ()), (C + "MANIFEST.sha256", ()), (C + "VERSION", ()),
    # 공통 설정 — 모델 이름·추론 강도
    (C + "model.json", ("common",)),
    # 주장 추출(공용 전처리)
    (C + "claims.json", ("extract",)), (C + "prompts/extract*.md", ("extract",)), ("app/judge/extract.py", ("extract",)),
    # 사실 일치
    (C + "degree.json", ("faithfulness",)), (C + "names.json", ("faithfulness",)),
    (C + "rule_constants.json", ("faithfulness",)), (C + "rules_calc.py", ("faithfulness",)),
    (C + "rules_degree.py", ("faithfulness",)), (C + "prompts/faithfulness.md", ("faithfulness",)),
    ("app/judge/faithfulness.py", ("faithfulness",)),
    # 숫자 대조 규칙은 사실 일치·일관성이 같이 쓴다
    (C + "rules_numbers.py", ("faithfulness", "consistency")),
    # 일관성
    (C + "contradictions.json", ("consistency",)), (C + "prompts/consistency.md", ("consistency",)),
    ("app/judge/consistency.py", ("consistency",)),
    # 필수 내용
    (C + "checklist.json", ("coverage",)), (C + "rules_conditions.py", ("coverage",)), (C + "prompts/coverage.md", ("coverage",)),
    ("app/judge/coverage.py", ("coverage",)),
    # 표기
    (C + "readability.json", ("readability",)), (C + "prompts/readability.md", ("readability",)),
    ("app/judge/readability.py", ("readability",)),
    # 합산·게이트·안전장치(코드)
    ("app/judge/aggregate.py", ("aggregate",)), ("app/judge/safety.py", ("aggregate",)),
    # 모든 모듈이 받는 것 — 읽는 범위 조각·재료·조회 자료·호출·설정·채점 순서, 생성기와 같이 쓰는 데이터 계산
    (C + "__init__.py", ("common",)), ("app/judge/reading.py", ("common",)), ("app/judge/material.py", ("common",)),
    ("app/judge/sources.py", ("common",)), ("app/judge/llm.py", ("common",)), ("app/judge/config.py", ("common",)),
    ("app/judge/core.py", ("common",)), ("app/narrative/derived.py", ("common",)), ("app/narrative/baselines.py", ("common",)),
    ("app/narrative/schemas.py", ("common",)),
    # 채점 밖 — 작업 흐름·사다리·알림·고객 묶음, 검증 계산(결과만 다시 계산)
    ("app/judge/ladder.py", ()), ("app/judge/alert.py", ()), ("app/judge/bundle.py", ()), ("app/judge/validation/*", ()),
    ("app/judge/__init__.py", ()),
    # 표에 없는 심판 코드·기준 파일 → 공통(보수적으로)
    (C + "*", ("common",)), ("app/judge/*", ("common",)),
)

# 출력을 받는 모듈: 추출 → 사실 일치·일관성. 네 축 → 합산.
DOWNSTREAM = {"extract": ("faithfulness", "consistency")}


def modules_of(path: str) -> tuple[str, ...]:
    path = path.strip().removeprefix("./")
    for pattern, mods in TABLE:
        if fnmatch(path, pattern):
            return mods
    return ()  # 저장소의 다른 곳(시험·문서·프론트·스크립트·서술 생성) — 이미 만든 서술의 채점에 닿지 않는다


def rescope(changed: list[str]) -> dict:
    """바뀐 파일 목록 → {modules, batches, human_axes, by_file}."""
    by_file = {p: list(modules_of(p)) for p in changed}
    direct = {m for mods in by_file.values() for m in mods}
    modules = set(direct)
    for m in direct:
        modules.update(DOWNSTREAM.get(m, ()))
    if "common" in modules or "extract" in modules:
        modules = {"common", "extract", *AXES, "aggregate"} if "common" in modules else modules | {"aggregate"}
        batches = list(ALL_BATCHES)
    elif modules:
        modules |= {"aggregate"} if modules & set(AXES) else set()
        batches = list(RUN_BATCHES)
    else:
        batches = []
    human = [a for a in AXES if a in modules]
    return {"modules": sorted(modules), "batches": batches, "human_axes": human, "by_file": by_file}
