"""tests/generate_s0_goldens.py — S0 고정 정답을 만드는 스크립트(손으로만 돌린다. pytest 가 모으지 않는다).

**정답을 다시 만드는 것은 경고 신호다.** 판 3 고정 테스트가 깨지면 먼저 코드를 고친다. 이 스크립트는 S0 에서 운영 main
코드로 정답을 한 번 뜨고, 그 뒤에는 "왜 바뀌었는지"를 사람이 확인한 예외에서만 쓴다(tests/golden_frozen/README.md).

쓰기 대상은 --out 으로 받은 빈 폴더뿐이다 — 저장소의 tests/golden_frozen 에 바로 쓰지 않는다. 결과를 눈으로 비교한 뒤
사람이 복사한다.

모드
- generate --out DIR : 지금 코드로 정답을 계산해 DIR 아래 golden_frozen 과 같은 상대 경로로 쓴다.
- extract --db C_VIEW_DB --requests C_REQ_DIR --out DIR : 판 3 발급본 DB(읽기 전용)에서 6건의 서술 칸 글을 뽑고,
  짝인 서술 요청 본문을 복사한다(tests/data/edition3_issued 와 같은 모양). 짝은 모델명·작업 유형·판정·facts 로 확인한다.

예: (저장소 루트에서) python tests/generate_s0_goldens.py generate --out /tmp/s0_out
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import frozen_s0 as fz  # noqa: E402
from app.main import app  # noqa: E402


def _write(out: Path, rel: str, text: str) -> None:
    path = out / rel
    if path.exists():
        raise SystemExit(f"이미 있음 — 덮어쓰지 않는다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate(out: Path) -> None:
    app.state.anthropic_client = None  # 옛 경로를 규칙 문장으로(키 없음)
    goldens: dict[str, object] = {}
    goldens.update(fz.legacy_route_goldens())
    goldens.update(asyncio.run(fz.job_fallback_goldens()))
    goldens.update(asyncio.run(fz.edition3_goldens()))
    goldens.update(fz.issued_goldens())
    for name, value in sorted(goldens.items()):
        _write(out, f"{name}.json", fz.dump(value))
    print(f"{len(goldens)}개 정답을 {out} 에 썼다")


def extract(db: Path, requests: Path, out: Path) -> None:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select r.report_no, r.model_name, s.content_json, s.content_hash from report r "
        "join report_snapshot s on s.report_id = r.id order by r.report_no"
    ).fetchall()
    seen = set()
    for report_no, model_name, content_json, content_hash in rows:
        if report_no not in fz.ISSUED_PAIRS:
            continue
        assert hashlib.sha256(content_json.encode("utf-8")).hexdigest() == content_hash, report_no
        content = json.loads(content_json)
        name = fz.ISSUED_PAIRS[report_no]
        req_text = (requests / f"{name}.request.json").read_text(encoding="utf-8")
        req = json.loads(req_text)
        # 짝 확인 — 파일 이름·작업 유형·모델명·판정·facts 가 모두 맞아야 한다.
        assert content["datasetInfo"]["fileName"] == f"{name}.csv", (report_no, name)
        assert content["meta"]["taskType"] == req["task_type"], (report_no, name)
        assert req["client_inputs"]["model"]["모델명"] == model_name, (report_no, model_name)
        assert content["conclusion"]["verdict"] == req["fact_sheet"]["verdict"], (report_no, name)
        assert content["narrativeFacts"] == req["fact_sheet"]["facts"], (report_no, name)
        assert content.get("contentSchema") == 3, report_no
        slots = {
            "report_no": report_no,
            "request": name,
            "task_type": req["task_type"],
            "model_name": model_name,
            "content_hash": content_hash,
            "narrative_source": content.get("narrativeSource"),
            "narrative_sections": content.get("narrativeSections"),
            "narrative_models": content.get("narrativeModels"),
            "parts": fz.slots_from_content(content),
        }
        _write(out, f"slots/{report_no}.json", json.dumps(slots, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write(out, f"requests/{name}.request.json", req_text)  # 바이트 그대로
        seen.add(report_no)
    assert seen == set(fz.ISSUED_PAIRS), seen
    print(f"발급본 {len(seen)}건을 {out} 에 뽑았다")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("extract")
    e.add_argument("--db", type=Path, required=True)
    e.add_argument("--requests", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve().is_relative_to(ROOT / "tests"):
        raise SystemExit("--out 은 저장소 tests/ 밖의 폴더여야 한다(정답을 바로 덮지 않는다)")
    if args.mode == "generate":
        generate(args.out)
    else:
        extract(args.db, args.requests, args.out)


if __name__ == "__main__":
    main()
