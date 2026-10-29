"""tests/test_judge_cost_report.py — 심판 비용·시간 집계(scripts/judge_cost_report.py), 심판 라운드 J9(계획 16-3).

합성 장부(열은 실행기 LEDGER_COLUMNS 와 같음)와 합성 작업 내부 기록으로 정확값을 본다. 진짜 장부·검증 폴더는 읽지 않는다.
"""
from __future__ import annotations

import csv
import json

import pytest

from scripts import judge_cost_report as R
from scripts.judge_live import LEDGER_COLUMNS


def _row(**kw) -> dict:
    row = {c: "" for c in LEDGER_COLUMNS}
    row.update({k: str(v) for k, v in kw.items()})
    return row


ROWS = [
    _row(run_id="R-1", stage="smoke", mode="live", seconds=2, cost_usd="0.001", input_tokens=10, output_tokens=2),
    _row(run_id="R-2", stage="pilot_generate", mode="live", seconds=30, cost_usd="0.200", input_tokens=1000, output_tokens=300),
    _row(run_id="R-3", stage="pilot_generate", mode="live", seconds=50, cost_usd="0.300", input_tokens=1200, output_tokens=400),
    _row(run_id="R-4", stage="pilot_generate", mode="live", seconds=1, cost_usd="0.000", error="api_error"),
    _row(run_id="R-5", stage="pilot_judge", mode="live", seconds=100, cost_usd="0.100", input_tokens=5000,
         cached_input_tokens=3000, output_tokens=700, reasoning_tokens=200),
    _row(run_id="R-6", stage="job_judge", mode="live", seconds=140, cost_usd="0.140", input_tokens=6000,
         cached_input_tokens=4000, output_tokens=900, reasoning_tokens=100),
    _row(run_id="R-7", stage="pilot_judge", mode="dev", seconds=9, cost_usd="", input_tokens=1),
]


def _write(tmp_path, rows) -> str:
    path = tmp_path / "장부.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def _call(purpose, seconds, cost, inp=100, cached=10, out=20, reason=5):
    return {"purpose": purpose, "seconds": seconds, "cost_usd": cost, "input_tokens": inp, "cached_input_tokens": cached,
            "output_tokens": out, "reasoning_tokens": reason}


def _round(n, secs: dict, cost, *, extract=(10.0, 5.0, 4.0)):
    record = {
        "seconds": {**secs, "total": round(sum(secs.values()), 3)},
        "extract": {"calls": [_call(f"extract_{i}", s, 0.001) for i, s in enumerate(extract)]},
        "faithfulness": {"calls": [_call("faithfulness", 1.0, 0.01), _call("faithfulness", 2.0, 0.02)]},
        "coverage": {"calls": [_call("coverage_1", 3.0, 0.005)]},
        "consistency": {"calls": [_call("consistency_1", 1.0, 0.002)]},
        "readability": {"calls": [_call("readability_1", 1.0, None)]},  # 비용 모름
    }
    return {"round": n, "seconds": record["seconds"]["total"], "cost_usd": cost, "record": record}


SECS1 = {"extract": 19.0, "faithfulness": 3.0, "coverage": 3.0, "consistency": 1.0, "readability": 1.0, "aggregate": 0.0}
SECS2 = {"extract": 21.0, "faithfulness": 5.0, "coverage": 4.0, "consistency": 2.0, "readability": 1.0, "aggregate": 0.0}

INTERNAL = {
    "seconds": 40.0, "cost_usd": 0.2, "seconds_total": 120.0,
    "judge": {"gate": "passed", "held_kind": None, "rounds": [_round(1, SECS1, 0.05), _round(2, SECS2, 0.07)],
              "summary": {"gate": "passed"}},
}
INTERNAL_HELD = {
    "seconds": 30.0, "cost_usd": None, "seconds_total": 60.0,
    "judge": {"gate": "held", "held_kind": "no_key", "rounds": []},
}


# ── 장부 ──


def test_ledger_by_kind_and_stage(tmp_path):
    rep = R.ledger_summary(R.read_ledger(_write(tmp_path, ROWS)), mode="live")
    assert rep["rows"] == 6 and rep["cost_usd"] == pytest.approx(0.741)
    gen = rep["by_kind"]["생성"]
    assert gen["rows"] == 3 and gen["seconds"]["sum"] == 81.0 and gen["cost_usd"] == pytest.approx(0.5)
    assert gen["ok_rows"] == 2 and gen["ok_seconds"]["mean"] == 40.0 and gen["ok_cost_mean_usd"] == pytest.approx(0.25)
    judge = rep["by_kind"]["채점"]
    assert judge["rows"] == 2 and judge["seconds"] == {"n": 2, "sum": 240.0, "mean": 120.0, "min": 100.0, "max": 140.0}
    assert judge["tokens"] == {"input_tokens": 11000, "cached_input_tokens": 7000, "output_tokens": 1600, "reasoning_tokens": 300}
    assert judge["ok_cost_mean_usd"] == pytest.approx(0.12)
    assert set(rep["by_stage"]) == {"smoke", "pilot_generate", "pilot_judge", "job_judge"}
    assert rep["by_kind"]["기타"]["rows"] == 1


def test_unknown_cost_makes_the_sum_unknown(tmp_path):
    rep = R.ledger_summary(R.read_ledger(_write(tmp_path, ROWS)))  # mode 거르지 않음 — dev 줄의 비용이 빈 칸
    assert rep["rows"] == 7 and rep["cost_usd"] is None
    assert rep["by_stage"]["pilot_judge"]["cost_usd"] is None
    assert rep["by_stage"]["pilot_judge"]["ok_cost_mean_usd"] is None
    assert rep["by_stage"]["job_judge"]["cost_usd"] == pytest.approx(0.14)


def test_sum_known():
    assert R.sum_known([]) == 0.0
    assert R.sum_known([0.1, 0.2]) == pytest.approx(0.3)
    assert R.sum_known([0.1, None]) is None


@pytest.mark.parametrize("stage, kind", [("pilot_judge", "채점"), ("job_judge", "채점"), ("verify_judge", "채점"),
                                         ("seal_generate", "생성"), ("smoke", "기타"), ("judge_pilot", "기타")])
def test_stage_kind(stage, kind):
    assert R.stage_kind(stage) == kind


# ── 작업 내부 기록 ──


def test_job_modules_and_extract_calls():
    rep = R.job_summary([INTERNAL, INTERNAL_HELD])
    assert rep["jobs"][0] == {"generation_seconds": 40.0, "generation_cost_usd": 0.2, "job_seconds": 120.0,
                              "judge_rounds": 2, "judge_seconds": 60.0, "judge_cost_usd": pytest.approx(0.12),
                              "gate": "passed", "held_kind": None}
    assert rep["jobs"][1]["judge_rounds"] == 0 and rep["jobs"][1]["judge_cost_usd"] == 0.0
    assert rep["generation_cost_usd"] is None  # 보류 작업의 생성 비용을 모름
    assert rep["judge_cost_usd"] == pytest.approx(0.12)
    ext = rep["modules"]["extract"]
    assert ext["seconds"] == {"n": 2, "sum": 40.0, "mean": 20.0, "min": 19.0, "max": 21.0}
    assert ext["calls"] == 6 and ext["cost_usd"] == pytest.approx(0.006)
    assert ext["tokens"] == {"input_tokens": 600, "cached_input_tokens": 60, "output_tokens": 120, "reasoning_tokens": 30}
    assert rep["modules"]["faithfulness"]["cost_usd"] == pytest.approx(0.06)
    assert rep["modules"]["faithfulness"]["seconds"]["sum"] == 8.0
    assert rep["modules"]["readability"]["cost_usd"] is None  # 호출 하나의 비용을 모름
    assert rep["modules"]["aggregate"]["calls"] == 0
    assert rep["extract_calls_seconds"]["extract_0"]["mean"] == 10.0
    assert rep["extract_calls_seconds"]["extract_2"]["n"] == 2
    assert rep["round_total_seconds"]["sum"] == 60.0


def test_job_file_shapes(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"tag": "x", "internal": INTERNAL}), encoding="utf-8")
    b = tmp_path / "b.json"
    b.write_text(json.dumps(INTERNAL), encoding="utf-8")
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"tag": "x"}), encoding="utf-8")
    assert R.read_job(a) == INTERNAL and R.read_job(b) == INTERNAL
    with pytest.raises(SystemExit):
        R.read_job(c)


def test_sealed_folder_is_never_read(tmp_path):
    sealed = tmp_path / "00_봉인"
    sealed.mkdir()
    (sealed / "x.json").write_text(json.dumps(INTERNAL), encoding="utf-8")
    (sealed / "장부.csv").write_text("run_id\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="봉인"):
        R.read_job(sealed / "x.json")
    with pytest.raises(SystemExit, match="봉인"):
        R.read_ledger(sealed / "장부.csv")


# ── 8절 효용성 표 ──


def test_efficacy_table_cells(tmp_path):
    rep = R.report(R.read_ledger(_write(tmp_path, ROWS)), [INTERNAL], mode="live", human_minutes=30)
    table = rep["efficacy"]
    assert list(table) == list(R.EFFICACY_ROWS) == ["판정 정확도", "사람 일치도", "비용·시간", "반복 가능"]
    assert all(list(v) == ["심판 없음 (사람만)", "심판 있음"] for v in table.values())
    assert table["비용·시간"]["심판 없음 (사람만)"] == "성적서 1건 채점에 인용 30분"
    assert table["비용·시간"]["심판 있음"] == (
        "채점 1회 평균 120초·$0.120 (장부 2회, 최대 140초); 성적서 1건(생성 + 채점 1회) 평균 약 160초·$0.370; "
        "작업 기록 1건: 작업 평균 120초, 채점 비용 합 $0.120")
    assert table["판정 정확도"]["심판 있음"] == R.FROM_VALIDATION
    md = R.efficacy_markdown(table)
    assert md.splitlines()[0] == "|  | 심판 없음 (사람만) | 심판 있음 |"
    assert "| 반복 가능 | 보고서마다 사람 동원 | 무제한 |" in md


def test_main_prints_markdown_and_json(tmp_path, capsys):
    ledger = _write(tmp_path, ROWS)
    job = tmp_path / "j.json"
    job.write_text(json.dumps(INTERNAL), encoding="utf-8")
    assert R.main(["--ledger", str(ledger), "--job", str(job), "--mode", "live"]) == 0
    out = capsys.readouterr().out
    assert "## 장부 — 6줄, 비용 합 $0.741" in out and "## 8절 효용성 표" in out and "| 사실 일치 | 2 |" in out
    assert R.main(["--ledger", str(ledger), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ledger"]["rows"] == 7 and data["jobs"] is None


def test_tool_does_not_import_the_app():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(R.__file__).read_text(encoding="utf-8"))
    mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith("app") for m in mods)


def test_sealed_folder_refused_in_other_spellings(tmp_path, monkeypatch):
    """자모 분해(NFD)·대소문자·링크로 적은 봉인 경로도 멈춘다(J10 검토 — judge_paths 한 곳)."""
    import unicodedata
    from pathlib import Path

    sealed = tmp_path / "검증" / "00_봉인"
    sealed.mkdir(parents=True)
    (sealed / "x.json").write_text("{}", encoding="utf-8")
    nfd = Path(unicodedata.normalize("NFD", str(sealed / "x.json")))
    assert "00_봉인" not in nfd.parts  # 글자 비교로는 못 잡는 철자
    for p in (nfd, Path(str(sealed / "x.json").replace("00_봉인", "00_봉인".upper()))):
        with pytest.raises(SystemExit, match="봉인"):
            R.read_job(p)
    link = tmp_path / "보통이름"
    link.symlink_to(sealed, target_is_directory=True)
    with pytest.raises(SystemExit, match="봉인"):
        R.read_job(link / "x.json")
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(tmp_path / "검증"))
    other = tmp_path / "검증" / "00_다른"
    other.mkdir()
    (other / "y.json").write_text(json.dumps({"judge": {}}), encoding="utf-8")
    R.read_job(other / "y.json")  # 봉인 밖은 읽는다
