"""tests/test_judge_annotation.py — 사람 주장 표시지·안내서(app/judge/validation/annotation.py, 계획 10-4)."""
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

from app.judge import criteria
from app.judge.validation import annotation
from tests.test_judge_live_runner import _RefusingClaude, _runner, jl

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "tests" / "data" / "judge_dev"


def _sealed(tmp_path) -> Path:
    req_dir = tmp_path / "요청"
    req_dir.mkdir()
    shutil.copy(DEV / "dev_multiclass.request.json", req_dir)
    runner, _, _ = _runner(tmp_path)
    jl.seal_generate(runner, req_dir, tmp_path / "봉인", {}, anthropic_client=_RefusingClaude())
    return tmp_path / "봉인"


def test_guide_comes_from_claims_json_and_reads_alone():
    text = annotation.guide_markdown()
    rules = criteria.load_json("claims.json")
    assert all(k["what"] in text for k in rules["kinds"]) and all(r in text for r in rules["unit_rules"])
    assert "설계" not in text and "6.1" not in text  # 원문을 가리키지 않는다


def test_sentences_cover_only_extract_scope(tmp_path):
    record = json.loads((_sealed(tmp_path) / "dev_multiclass.json").read_text(encoding="utf-8"))
    pieces = annotation.pieces_from_sealed(record)
    rows = annotation.sentence_rows(pieces)
    assert rows and [r[0] for r in rows] == list(range(1, len(rows) + 1))
    assert {r[4] for r in rows} <= {"ai", "template"}
    assert any(p.source in ("fixed", "table") for p in pieces)  # 범위 밖 조각은 있었지만 표시지에 없다
    by_piece = {}
    for r in rows:
        by_piece.setdefault(r[1], []).append(r[5])
    for p in pieces:
        if p.id in by_piece:
            assert all(s in p.text for s in by_piece[p.id])


def test_script_prints_counts_only_and_keeps_filled_claims(tmp_path):
    sealed = _sealed(tmp_path)
    out = tmp_path / "정답지"
    out.mkdir()
    (out / "dev_multiclass.claims.csv").write_text("문장번호,종류,인용 구절,메모\n1,number,0.335,\n", encoding="utf-8")
    run = subprocess.run([sys.executable, "scripts/judge_annotate.py", "--sealed", str(sealed), "--out", str(out)],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(run.stdout)
    assert [r["tag"] for r in report] == ["dev_multiclass"] and report[0]["sentences"] > 0
    record = json.loads((sealed / "dev_multiclass.json").read_text(encoding="utf-8"))
    narrative = record["view"]["result"]["conclusion"]["narrative"]
    assert narrative and narrative[:20] not in run.stdout  # 서술 글을 찍지 않는다
    assert "0.335" in (out / "dev_multiclass.claims.csv").read_text(encoding="utf-8")  # 덮어쓰지 않았다
    head = next(csv.reader(io.StringIO((out / "dev_multiclass.sentences.csv").read_text(encoding="utf-8"))))
    assert tuple(head) == annotation.SENTENCE_COLUMNS
    assert (out / "표시_안내서.md").exists()
