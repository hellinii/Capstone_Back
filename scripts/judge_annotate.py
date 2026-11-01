"""
judge_annotate.py — 봉인 서술의 주장 표시지·안내서를 만든다(계획 10-4). 서술 글을 화면에 찍지 않는다(개수·해시만).

레포 루트에서: python scripts/judge_annotate.py --sealed <검증/00_봉인> --out <검증/01_정답지>
이미 채워 둔 claims.csv 는 덮어쓰지 않는다.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.judge.validation import annotation  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sealed, out = Path(args.sealed), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "표시_안내서.md").write_text(annotation.guide_markdown(), encoding="utf-8")
    report = []
    for path in sorted(sealed.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rows = annotation.sentence_rows(annotation.pieces_from_sealed(record))
        text = annotation.to_csv(annotation.SENTENCE_COLUMNS, rows)
        (out / f"{record['tag']}.sentences.csv").write_text(text, encoding="utf-8")
        claims = out / f"{record['tag']}.claims.csv"
        if not claims.exists():
            claims.write_text(annotation.to_csv(annotation.CLAIM_COLUMNS, []), encoding="utf-8")
        report.append({"tag": record["tag"], "sentences": len(rows), "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
