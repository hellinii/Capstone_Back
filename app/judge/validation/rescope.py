"""app/judge/validation/rescope.py — 재검증 범위 계산(계획 정10, 설계 7.7 "다시 도는 범위"). 순수 함수.

규칙(정10): 바뀐 모듈과 그 출력을 받는 모듈을 모든 실행(재현성·결함·sham)에서 다시 돈다. 주장 추출이나 공통 설정(모델·추론 강도)을
고치면 4종 전부(추출 재현율·정밀도까지). 바뀐 축의 LLM 판정이 달라졌으면 그 축의 사람 일치도 표본도 다시 채점한다.

채점 한 번(judge_once)은 늘 네 축을 함께 돌리므로 "다시 돌 모듈"은 다시 비교할 결과의 범위이고, 다시 부르는 채점 횟수는 묶음
회차 그대로다. 표에 없는 심판 코드(app/judge 아래 새 파일)는 모르는 것이라 공통으로 본다(덜 돌리는 쪽으로 틀리지 않게).

표 밖 app/ 파일(J10 검토): 채점 입구(app/judge/core.py — judge_once)에서 import 를 따라 닿는 app/ 파일은 채점에 들어가는 코드다
(예: app/narrative/wording_ed4.py — 읽는 범위 조각의 고정 문구). 표에 줄이 없으면 공통으로 본다. import 그래프는 소스를 읽어(ast)
계산한다 — 함수 안의 늦은 import·상대 import·패키지 __init__ 까지. 표에 줄이 있는 파일은 표가 먼저다.

상호작용
- 의존(import): fnmatch, ast, functools, pathlib
- 사용처: scripts/judge_validate.py rescope, 시험 tests/test_judge_validation_rescope.py
"""
from __future__ import annotations

import ast
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JUDGE_ENTRIES = ("app/judge/core.py",)  # 채점 한 번(judge_once)의 입구

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
    # 1.1 보탬 — 추론 표시·규범·범위 낱말(추출이 걸러내고 사실 일치가 어미를 본다), 칸 설명(추출 칸 목록과 조회 답에 실린다)
    (C + "inference_rules.json", ("extract", "faithfulness")), (C + "field_notes.json", ("extract", "faithfulness")),
    # 사실 일치
    (C + "degree.json", ("faithfulness",)), (C + "names.json", ("faithfulness",)),
    (C + "rule_constants.json", ("faithfulness",)), (C + "rules_calc.py", ("faithfulness",)),
    (C + "rules_degree.py", ("faithfulness",)), (C + "prompts/faithfulness.md", ("faithfulness",)),
    (C + "prompts/faithfulness_recheck.md", ("faithfulness",)), (C + "remap_anchors.json", ("faithfulness",)),
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
    # 검증 기록 잠금(J9): 판 4 작업이 심판을 부를지 정하는 운영 장치 — judge_once 의 판정·점수에 닿지 않는다. 기록 파일도 같다.
    ("app/judge/criteria_lock.py", ()), ("app/judge/validated.json", ()),
    # 표에 없는 심판 코드·기준 파일 → 공통(보수적으로)
    (C + "*", ("common",)), ("app/judge/*", ("common",)),
)

# 출력을 받는 모듈: 추출 → 사실 일치·일관성. 네 축 → 합산.
DOWNSTREAM = {"extract": ("faithfulness", "consistency")}


def _module_file(root: Path, dotted: str) -> str | None:
    """app.x.y → 'app/x/y.py' 또는 'app/x/y/__init__.py'(있는 것). app 밖이면 None."""
    if dotted != "app" and not dotted.startswith("app."):
        return None
    base = root / Path(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand.relative_to(root).as_posix()
    return None


def _imports_of(root: Path, rel: str) -> set[str]:
    """한 파일이 import 하는 app/ 파일들(패키지 __init__ 포함)."""
    path = root / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    package = rel.removesuffix(".py").replace("/", ".")
    package = package.removesuffix(".__init__") if rel.endswith("__init__.py") else package.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1]) if node.level <= len(parts) else ""
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)  # from pkg import 모듈
    out: set[str] = set()
    for name in names:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):  # 패키지 __init__ 들도 실행된다
            f = _module_file(root, ".".join(parts[:i]))
            if f:
                out.add(f)
    return out


@lru_cache(maxsize=4)
def judge_import_closure(root: Path = REPO) -> frozenset[str]:
    """채점 입구(JUDGE_ENTRIES)에서 import 를 따라 닿는 app/ 파일 전부(입구 포함)."""
    seen: set[str] = set()
    todo = [e for e in JUDGE_ENTRIES if (root / e).is_file()]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        todo.extend(_imports_of(root, rel) - seen)
    return frozenset(seen)


def modules_of(path: str, *, root: Path = REPO) -> tuple[str, ...]:
    path = path.strip().removeprefix("./")
    for pattern, mods in TABLE:
        if fnmatch(path, pattern):
            return mods
    if path in judge_import_closure(root):
        return ("common",)  # 표에 없어도 채점이 import 하는 앱 코드 — 공통(보수적으로)
    return ()  # 저장소의 다른 곳(시험·문서·프론트·스크립트·채점이 쓰지 않는 서술 생성) — 이미 만든 서술의 채점에 닿지 않는다


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
