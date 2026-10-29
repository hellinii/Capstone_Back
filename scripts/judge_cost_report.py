"""scripts/judge_cost_report.py — 심판 비용·시간 집계(심판 라운드 J9, 계획 16-3 · 설계 5.5 "대기 시간과 비용"·8절).

두 재료를 읽어 단계별 시간·토큰·비용을 모으고, 8절 효용성 표의 칸을 채운다. 읽기만 한다(장부·결과 파일에 쓰지 않는다).

1. 장부(`검증/장부.csv`, 열은 `검증/장부_열.txt`) — 한 줄 = 실호출 한 번. 단계(stage) 이름 끝으로 나눈다: `_judge` = 채점 1회,
   `_generate` = 서술 생성 1건, `validate_<묶음>`(검증 실행기 — 재현성·결함·sham 회차) = 채점 1회, 그 밖은 기타(스모크 등).
2. 작업 내부 기록(`internal_json`, 선택 `--job`) — 서술 작업 하나의 생성 시간·비용과 심판 기록(internal["judge"]): 채점 회차마다
   모듈별 시간(전처리·사실 일치·필수 내용·일관성·표기·합산), 전처리 0·1·2번 호출 시간, 모듈별 토큰·비용.
   파일은 내부 기록 dict 그대로이거나 `{"internal": {...}}`(scripts/judge_live.py job 결과 꼴)이다.

비용 규칙(지금 앱과 같게): 비용을 모르는 줄(빈 칸)이 하나라도 있으면 그 합은 None — 추정하지 않는다.
봉인 폴더(`00_봉인`)의 파일은 읽지 않는다(계획 §카-3) — 그런 경로를 주면 멈춘다.

    python scripts/judge_cost_report.py --ledger ../JUDGE_ROUND/검증/장부.csv [--job 결과.json ...] [--mode live]
        [--human-minutes 30] [--json]

상호작용
- 의존(import): 표준 라이브러리만(csv, json, argparse) + scripts/judge_paths(경로 비교, 표준 라이브러리만) — 앱을 import 하지 않는다
- 사용처: 사람(V2 끝 8절 효용성 표 채우기), tests/test_judge_cost_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_paths  # noqa: E402 — 경로 비교 한 곳(표준 라이브러리만)

SEALED = "00_봉인"
TOKEN_COLUMNS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
# 심판 기록의 모듈 이름 → 표에 쓰는 이름(설계 5.5·계획 16-1 의 단계). 순서도 이것.
MODULES = (("extract", "전처리(주장 추출)"), ("faithfulness", "사실 일치"), ("coverage", "필수 내용"),
           ("consistency", "일관성"), ("readability", "표기"), ("aggregate", "합산"))
EXTRACT_CALLS = ("extract_0", "extract_1", "extract_2")

# 설계 8절 효용성 표 — 칸 이름 그대로.
EFFICACY_HEAD = ("", "심판 없음 (사람만)", "심판 있음")
EFFICACY_ROWS = ("판정 정확도", "사람 일치도", "비용·시간", "반복 가능")
FROM_VALIDATION = "(7절 검증 결과에서 — 이 도구가 채우지 않음)"


# ── 작은 셈 ──


def _num(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def sum_known(values) -> float | None:
    """하나라도 모르면(None) None. 빈 목록은 0."""
    total = 0.0
    for v in values:
        if v is None:
            return None
        total += v
    return round(total, 6)


def spread(values: list[float]) -> dict:
    """시간 묶음의 합·평균·최소·최대(값이 없으면 모두 None)."""
    if not values:
        return {"n": 0, "sum": None, "mean": None, "min": None, "max": None}
    return {"n": len(values), "sum": round(sum(values), 3), "mean": round(statistics.fmean(values), 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


VALIDATE_PREFIX = "validate_"  # 검증 실행기(scripts/judge_validate.py) 장부 단계 — validate_<묶음>, 한 줄 = 채점 1회


def stage_kind(stage: str) -> str:
    if stage.endswith("_judge") or stage.startswith(VALIDATE_PREFIX):
        return "채점"
    if stage.endswith("_generate"):
        return "생성"
    return "기타"


def _refuse_sealed(path: Path) -> None:
    """봉인 폴더 경로면 멈춘다 — 마디 이름(자모 분해·대소문자·링크까지 judge_paths 로)과, 검증 폴더를 알면 그 00_봉인 안인지."""
    vdir = os.getenv("JUDGE_VALIDATION_DIR", "").strip()
    if judge_paths.has_part(path, SEALED) or (vdir and judge_paths.inside(path, Path(vdir) / SEALED)):
        raise SystemExit(f"봉인 폴더의 파일은 읽지 않는다(계획 §카-3): {path}")


# ── 장부 ──


def read_ledger(path: Path) -> list[dict]:
    _refuse_sealed(path)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ledger_summary(rows: list[dict], *, mode: str | None = None) -> dict:
    """단계별·종류별(채점·생성·기타) 줄 수·시간·토큰·비용."""
    if mode:
        rows = [r for r in rows if (r.get("mode") or "") == mode]

    def block(group: list[dict]) -> dict:
        secs = [s for s in (_num(r.get("seconds")) for r in group) if s is not None]
        costs = [_num(r.get("cost_usd")) for r in group]
        # 평균(8절 표)은 오류 줄(error 열이 찬 줄 — 크레딧 부족·API 오류로 1초 만에 끝난 생성 등)을 뺀 줄로 낸다. 합은 모든 줄.
        ok = [r for r in group if not (r.get("error") or "").strip()]
        ok_secs = [s for s in (_num(r.get("seconds")) for r in ok) if s is not None]
        ok_costs = [_num(r.get("cost_usd")) for r in ok]
        return {
            "rows": len(group),
            "seconds": spread(secs),
            "tokens": {c: int(sum(_num(r.get(c)) or 0 for r in group)) for c in TOKEN_COLUMNS},
            "cost_usd": sum_known(costs),
            "ok_rows": len(ok),
            "ok_seconds": spread(ok_secs),
            "ok_cost_mean_usd": None if not ok_costs or None in ok_costs else round(statistics.fmean(ok_costs), 6),
        }

    stages: dict[str, list[dict]] = {}
    kinds: dict[str, list[dict]] = {}
    for r in rows:
        stage = r.get("stage") or ""
        stages.setdefault(stage, []).append(r)
        kinds.setdefault(stage_kind(stage), []).append(r)
    return {
        "rows": len(rows),
        "cost_usd": sum_known(_num(r.get("cost_usd")) for r in rows),
        "by_stage": {s: block(g) for s, g in sorted(stages.items())},
        "by_kind": {k: block(g) for k, g in sorted(kinds.items())},
    }


# ── 작업 내부 기록 ──


def read_job(path: Path) -> dict:
    _refuse_sealed(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("internal"), dict):
        return data["internal"]
    if isinstance(data, dict) and "judge" in data:
        return data
    raise SystemExit(f"작업 내부 기록이 아니다(internal_json 또는 {{\"internal\": ...}}): {path}")


def _calls_of(record: dict, module: str) -> list[dict]:
    part = record.get(module)
    return list(part.get("calls") or []) if isinstance(part, dict) else []


def job_summary(internals: list[dict]) -> dict:
    """작업마다 생성·채점 시간과 비용, 그리고 모든 채점 회차를 모은 모듈별 시간·토큰·비용."""
    jobs = []
    module_secs: dict[str, list[float]] = {m: [] for m, _ in MODULES}
    module_total: list[float] = []
    extract_secs: dict[str, list[float]] = {p: [] for p in EXTRACT_CALLS}
    module_calls: dict[str, list[dict]] = {m: [] for m, _ in MODULES}
    for internal in internals:
        judge = internal.get("judge") or {}
        rounds = [r for r in judge.get("rounds") or [] if isinstance(r.get("record"), dict)]
        summary = judge.get("summary") or {}
        judge_costs = [_num(r.get("cost_usd")) for r in rounds]
        jobs.append({
            "generation_seconds": _num(internal.get("seconds")),
            "generation_cost_usd": _num(internal.get("cost_usd")),
            "job_seconds": _num(internal.get("seconds_total")),
            "judge_rounds": len(rounds),
            "judge_seconds": round(sum(_num(r.get("seconds")) or 0 for r in rounds), 3),
            "judge_cost_usd": sum_known(judge_costs),
            "gate": judge.get("gate") or summary.get("gate"),
            "held_kind": judge.get("held_kind"),
        })
        for r in rounds:
            record = r["record"]
            secs = record.get("seconds") or {}
            for m, _ in MODULES:
                v = _num(secs.get(m))
                if v is not None:
                    module_secs[m].append(v)
                module_calls[m].extend(_calls_of(record, m))
            total = _num(secs.get("total"))
            if total is not None:
                module_total.append(total)
            for c in _calls_of(record, "extract"):
                v = _num(c.get("seconds"))
                if c.get("purpose") in extract_secs and v is not None:
                    extract_secs[c["purpose"]].append(v)
    modules = {}
    for m, name in MODULES:
        calls = module_calls[m]
        modules[m] = {
            "name": name,
            "seconds": spread(module_secs[m]),
            "calls": len(calls),
            "tokens": {c: int(sum(_num(x.get(c)) or 0 for x in calls)) for c in TOKEN_COLUMNS},
            "cost_usd": sum_known(_num(x.get("cost_usd")) for x in calls),
        }
    return {
        "jobs": jobs,
        "modules": modules,
        "round_total_seconds": spread(module_total),
        "extract_calls_seconds": {p: spread(v) for p, v in extract_secs.items()},
        "judge_cost_usd": sum_known(j["judge_cost_usd"] for j in jobs),
        "generation_cost_usd": sum_known(j["generation_cost_usd"] for j in jobs),
    }


# ── 8절 효용성 표 ──


def _money(v: float | None) -> str:
    return "모름" if v is None else f"${v:.3f}"


def _secs(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}초"


def efficacy_table(ledger: dict, jobs: dict | None, *, human_minutes: float | None = None) -> dict:
    """8절 표의 칸(행 → {열 → 글}). 이 도구가 채우는 칸은 '비용·시간' 둘과 '반복 가능'(설계 글 그대로)."""
    judge = ledger["by_kind"].get("채점")
    gen = ledger["by_kind"].get("생성")
    parts = []
    if judge and judge["ok_rows"]:
        parts.append(f"채점 1회 평균 {_secs(judge['ok_seconds']['mean'])}·{_money(judge['ok_cost_mean_usd'])} "
                     f"(장부 {judge['ok_rows']}회, 최대 {_secs(judge['ok_seconds']['max'])})")
    if judge and gen and judge["ok_rows"] and gen["ok_rows"]:
        js, gs = judge["ok_seconds"]["mean"], gen["ok_seconds"]["mean"]
        jc, gc = judge["ok_cost_mean_usd"], gen["ok_cost_mean_usd"]
        mean_s = None if None in (js, gs) else js + gs
        mean_c = None if None in (jc, gc) else round(jc + gc, 6)
        parts.append(f"성적서 1건(생성 + 채점 1회) 평균 약 {_secs(mean_s)}·{_money(mean_c)}")
    if jobs and jobs["jobs"]:
        n = len(jobs["jobs"])
        totals = [j["job_seconds"] for j in jobs["jobs"] if j["job_seconds"] is not None]
        parts.append(f"작업 기록 {n}건: 작업 평균 {_secs(statistics.fmean(totals) if totals else None)}, "
                     f"채점 비용 합 {_money(jobs['judge_cost_usd'])}")
    with_judge = "; ".join(parts) if parts else "(장부에 채점 줄 없음)"
    without = f"성적서 1건 채점에 인용 {human_minutes:g}분" if human_minutes is not None else "성적서 1건 채점에 인용 N분 (사람 평가 시간 기록에서)"
    return {
        "판정 정확도": {"심판 없음 (사람만)": FROM_VALIDATION, "심판 있음": FROM_VALIDATION},
        "사람 일치도": {"심판 없음 (사람만)": FROM_VALIDATION, "심판 있음": FROM_VALIDATION},
        "비용·시간": {"심판 없음 (사람만)": without, "심판 있음": with_judge},
        "반복 가능": {"심판 없음 (사람만)": "보고서마다 사람 동원", "심판 있음": "무제한"},
    }


def efficacy_markdown(table: dict) -> str:
    lines = ["| " + " | ".join(EFFICACY_HEAD) + " |", "|---|---|---|"]
    for row in EFFICACY_ROWS:
        lines.append(f"| {row} | {table[row][EFFICACY_HEAD[1]]} | {table[row][EFFICACY_HEAD[2]]} |")
    return "\n".join(lines)


def report(ledger_rows: list[dict], internals: list[dict], *, mode: str | None = None,
           human_minutes: float | None = None) -> dict:
    ledger = ledger_summary(ledger_rows, mode=mode)
    jobs = job_summary(internals) if internals else None
    return {"ledger": ledger, "jobs": jobs, "efficacy": efficacy_table(ledger, jobs, human_minutes=human_minutes)}


def markdown(rep: dict) -> str:
    led = rep["ledger"]
    out = [f"## 장부 — {led['rows']}줄, 비용 합 {_money(led['cost_usd'])}", "",
           "| 단계 | 줄 | 시간 합 | 평균 | 최대 | 입력 토큰 | 캐시 | 출력 | 추론 | 비용 합 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label, block in [*((f"{k}(종류)", v) for k, v in led["by_kind"].items()), *led["by_stage"].items()]:
        s, t = block["seconds"], block["tokens"]
        out.append(f"| {label} | {block['rows']} | {_secs(s['sum'])} | {_secs(s['mean'])} | {_secs(s['max'])} | "
                   f"{t['input_tokens']} | {t['cached_input_tokens']} | {t['output_tokens']} | {t['reasoning_tokens']} | "
                   f"{_money(block['cost_usd'])} |")
    jobs = rep["jobs"]
    if jobs:
        out += ["", f"## 작업 내부 기록 — {len(jobs['jobs'])}건", "",
                "| 모듈 | 회차 | 시간 합 | 평균 | 최대 | 호출 | 입력 토큰 | 캐시 | 출력 | 추론 | 비용 합 |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, block in jobs["modules"].items():
            s, t = block["seconds"], block["tokens"]
            out.append(f"| {block['name']} | {s['n']} | {_secs(s['sum'])} | {_secs(s['mean'])} | {_secs(s['max'])} | "
                       f"{block['calls']} | {t['input_tokens']} | {t['cached_input_tokens']} | {t['output_tokens']} | "
                       f"{t['reasoning_tokens']} | {_money(block['cost_usd'])} |")
        out.append("")
        out.append("전처리 호출별 평균: " + ", ".join(
            f"{p[-1]}번 {_secs(v['mean'])}" for p, v in jobs["extract_calls_seconds"].items()))
        out.append(f"채점 회차 합계 시간: 평균 {_secs(jobs['round_total_seconds']['mean'])}, "
                   f"최대 {_secs(jobs['round_total_seconds']['max'])}")
    out += ["", "## 8절 효용성 표", "", efficacy_markdown(rep["efficacy"])]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="심판 비용·시간 집계(8절 효용성 표)")
    ap.add_argument("--ledger", required=True, type=Path, help="장부.csv")
    ap.add_argument("--job", action="append", default=[], type=Path, help="작업 내부 기록 JSON(여러 번)")
    ap.add_argument("--mode", default=None, help="장부 mode 열로 거름(예 live)")
    ap.add_argument("--human-minutes", type=float, default=None, help="사람이 성적서 1건 채점에 쓴 분(8절 '심판 없음' 칸)")
    ap.add_argument("--json", action="store_true", help="JSON 으로 찍기")
    args = ap.parse_args(argv)
    rep = report(read_ledger(args.ledger), [read_job(p) for p in args.job], mode=args.mode,
                 human_minutes=args.human_minutes)
    sys.stdout.write(json.dumps(rep, ensure_ascii=False, indent=2) + "\n" if args.json else markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
