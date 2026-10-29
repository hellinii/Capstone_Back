"""재검증 범위(계획 정10, app/judge/validation/rescope.py) — 바뀐 파일 → 다시 돌 모듈·묶음 표.

- 추출·공통 설정(model.json)이면 4종 전부, 한 축이면 그 축 + 합산을 실행 묶음 셋(재현성·결함·sham)에서.
- 기준 묶음의 모든 파일과 app/judge 의 모든 .py 가 표에 제 줄을 갖는다(새 파일을 더하면 표를 고치게 — 보수적 대체 줄에 숨지 않게).
"""
from __future__ import annotations

import importlib.util
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

from app.judge import criteria
from app.judge.validation import rescope as RS

ROOT = Path(__file__).resolve().parents[1]
ALL4 = ["reproducibility", "defects", "sham", "extraction"]
RUN3 = ["reproducibility", "defects", "sham"]
EVERY = ["aggregate", "common", "consistency", "coverage", "extract", "faithfulness", "readability"]


@pytest.mark.parametrize("changed, modules, batches, human", [
    (["app/judge/criteria/model.json"], EVERY, ALL4, ["faithfulness", "consistency", "coverage", "readability"]),
    (["app/judge/criteria/claims.json"], ["aggregate", "consistency", "extract", "faithfulness"], ALL4,
     ["faithfulness", "consistency"]),
    (["app/judge/criteria/prompts/extract_self_check.md"], ["aggregate", "consistency", "extract", "faithfulness"], ALL4,
     ["faithfulness", "consistency"]),
    (["app/judge/core.py"], EVERY, ALL4, ["faithfulness", "consistency", "coverage", "readability"]),
    (["app/judge/criteria/checklist.json"], ["aggregate", "coverage"], RUN3, ["coverage"]),
    (["app/judge/criteria/prompts/readability.md"], ["aggregate", "readability"], RUN3, ["readability"]),
    (["app/judge/criteria/contradictions.json"], ["aggregate", "consistency"], RUN3, ["consistency"]),
    (["app/judge/criteria/degree.json"], ["aggregate", "faithfulness"], RUN3, ["faithfulness"]),
    (["app/judge/criteria/rules_numbers.py"], ["aggregate", "consistency", "faithfulness"], RUN3, ["faithfulness", "consistency"]),
    (["app/judge/aggregate.py"], ["aggregate"], RUN3, []),
    (["app/judge/criteria/prompts/coverage.md", "app/judge/readability.py"],
     ["aggregate", "coverage", "readability"], RUN3, ["coverage", "readability"]),
    (["app/judge/criteria/CHANGELOG.md", "app/judge/criteria/VERSION", "app/judge/criteria/VERSIONS.json"], [], [], []),
    (["app/judge/ladder.py", "app/judge/validation/stats.py", "tests/test_x.py", "docs/ARCHITECTURE.md"], [], [], []),
    (["app/judge/새모듈.py"], EVERY, ALL4, ["faithfulness", "consistency", "coverage", "readability"]),  # 모르는 심판 코드
])
def test_rescope_table(changed, modules, batches, human):
    got = RS.rescope(changed)
    assert (got["modules"], got["batches"], got["human_axes"]) == (modules, batches, human)


def _explicit(path: str) -> bool:
    fallback = {RS.C + "*", "app/judge/*"}
    return any(fnmatch(path, pat) for pat, _ in RS.TABLE if pat not in fallback)


def test_every_criteria_file_has_its_own_row():
    files = [f"app/judge/criteria/{p.relative_to(criteria.HERE).as_posix()}" for p in criteria.bundle_files()]
    files += [RS.C + n for n in ("CHANGELOG.md", "VERSIONS.json", "MANIFEST.sha256")]
    assert [f for f in files if not _explicit(f)] == []


def test_every_judge_module_has_its_own_row():
    files = [p.relative_to(ROOT).as_posix() for p in (ROOT / "app" / "judge").glob("*.py")]
    assert len(files) > 10
    assert [f for f in files if not _explicit(f)] == []


def test_cli_rescope_prints_and_uses_git_since(capsys):
    spec = importlib.util.spec_from_file_location("judge_validate", ROOT / "scripts" / "judge_validate.py")
    jv = sys.modules.get("judge_validate")
    if jv is None:
        jv = importlib.util.module_from_spec(spec)
        sys.modules["judge_validate"] = jv
        spec.loader.exec_module(jv)
    lines: list[str] = []
    got = jv.cmd_rescope([], "judge-criteria-v1.0",
                         git=lambda *a: (0, "app/judge/criteria/checklist.json\n") if a[0] == "diff" else (1, ""),
                         echo=lines.append)
    assert got["batches"] == RUN3 and lines[0] == "다시 돌 모듈: aggregate, coverage"
    with pytest.raises(jv.RunnerError, match="바뀐 파일이 없다"):
        jv.cmd_rescope([], None, echo=lines.append)
