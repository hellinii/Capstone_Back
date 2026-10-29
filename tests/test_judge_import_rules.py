"""tests/test_judge_import_rules.py — 심판과 서술 생성 사이의 import 규칙(심판 라운드 J1 ④, 계획 사-2).

1. 서술 생성 지시문 쪽 모듈은 app.judge 를 import 하지 않는다 — 지시문을 만드는 쪽이 심판 결과(사유·인용)를 알 길을 막는다
   (심판 설계 5.5 "심판의 사유는 프롬프트에 넣지 않는다"). 작업 흐름(app.narrative.jobs)만 심판을 부른다.
2. app.judge 는 서술 쪽 판정 규칙(checks·grounding·required_items)을 import 하지 않는다 — 서술 검문이 바뀌면 심판 기준이 몰래
   바뀌어 기준 고정(7.5)이 깨진다. 서술 쪽 데이터(schemas·sections·generator 의 GenerationSession)는 써도 된다.
3. app 은 scripts 를 import 하지 않는다(가짜 LLM 개발 서버·실호출 실행기가 운영 경로에 끼지 않게).
4. 검증 도구(app.judge.validation)는 앱 실행 경로에서 import 되지 않는다 — validation 밖의 app 모듈은 그것을 import 하지 않고,
   app.main 을 띄워도 sys.modules 에 실리지 않는다(운영 서버에 검증 코드가 실리지 않게, 계획 17-2).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"

PROMPT_SIDE = ["prompt", "generator", "checks", "grounding", "required_items", "fallback", "fact_view", "fact_view_ed4",
               "wording", "wording_ed4"]
NARRATIVE_RULES = ("app.narrative.checks", "app.narrative.grounding", "app.narrative.required_items")


def imported_modules(path: Path) -> set[str]:
    """파일이 import 하는 모듈 이름 전부(from a import b 는 a 와 a.b 둘 다 — 'from app import judge' 도 잡히게)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _hits(names: set[str], prefixes) -> list[str]:
    return sorted(n for n in names if any(n == p or n.startswith(p + ".") for p in prefixes))


def test_prompt_side_modules_do_not_import_judge():
    files = [APP / "narrative" / f"{m}.py" for m in PROMPT_SIDE]
    assert all(f.exists() for f in files), [f.name for f in files if not f.exists()]
    bad = {f.name: _hits(imported_modules(f), ["app.judge"]) for f in files}
    assert {k: v for k, v in bad.items() if v} == {}


def test_judge_does_not_import_narrative_rules():
    files = sorted((APP / "judge").rglob("*.py"))
    assert len(files) >= 3
    bad = {str(f.relative_to(APP)): _hits(imported_modules(f), NARRATIVE_RULES) for f in files}
    assert {k: v for k, v in bad.items() if v} == {}


def test_app_does_not_import_scripts():
    files = sorted(APP.rglob("*.py"))
    assert len(files) > 50
    bad = {str(f.relative_to(APP)): _hits(imported_modules(f), ["scripts"]) for f in files}
    assert {k: v for k, v in bad.items() if v} == {}


VALIDATION = APP / "judge" / "validation"


def test_app_runtime_does_not_import_validation():
    files = [f for f in sorted(APP.rglob("*.py")) if VALIDATION not in f.parents]
    assert len(files) > 50
    bad = {str(f.relative_to(APP)): _hits(imported_modules(f), ["app.judge.validation"]) for f in files}
    assert {k: v for k, v in bad.items() if v} == {}


def test_app_main_does_not_load_validation():
    # 실제로 띄워 본다 — 동적 import·__init__ 재수출로 끼어들어도 잡힌다.
    code = ("import sys, json, app.main; "
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('app.'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert "app.judge.core" in loaded  # 이빨: 심판 본체는 실린다(검사가 헛돌지 않음)
    assert [m for m in loaded if m == "app.judge.validation" or m.startswith("app.judge.validation.")] == []


def test_detector_sees_every_import_form(tmp_path):
    # 이빨: 세 꼴 모두 잡는다.
    src = tmp_path / "x.py"
    src.write_text("import app.judge.llm\nfrom app import judge\nfrom app.narrative.checks import check_section\n", encoding="utf-8")
    names = imported_modules(src)
    assert _hits(names, ["app.judge"]) == ["app.judge", "app.judge.llm"]
    assert _hits(names, NARRATIVE_RULES) == ["app.narrative.checks", "app.narrative.checks.check_section"]
