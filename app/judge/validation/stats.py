"""app/judge/validation/stats.py — 심판 검증 통계 순수 함수(계획 17-2, 설계 7.1·7.6, 질문 Q10·Q11).

- ICC(A,1): 두 방향 무작위, 절대 일치, 단일 측정(McGraw & Wong 1996 의 ICC(A,1) = Shrout & Fleiss 1979 의 ICC(2,1)) +
  95% 신뢰구간(McGraw & Wong 1996 Case 2A 식 — Satterthwaite 자유도 v 와 F 분포 분위).
- 분산 0 규칙(Q10 추천): 한 축에서 모든 성적서의 모든 회차 점수가 같으면 그 축은 통과. ICC 는 None, 이유는 note 에.
- 축별 점수(Q10 추천): 사실 일치 부합 비율, 필수 내용 서술 비율, 일관성 모순 건수, 표기 위반 건수 — judge_once 의 scorecard
  (app.judge.aggregate.scorecard) 키에서 읽는다. 키가 없거나 분모가 0 이면 None.
- 게이트 재현성: 같은 성적서 반복에서 게이트 판정이 모두 같음(실패 회차 None 은 다른 판정으로 본다).
- 결함 탐지·독립성·sham(Q11 추천): 탐지 = 심은 문장을 짚은 판정(사실 일치·일관성·표기는 그 문장을 인용한 위반, 필수 내용은
  지운 항목이 "없음"). 흔들림 폭 = 바탕 성적서 자신의 재현성 범위(최대 − 최소). 독립성·sham = 회차 요약값(평균 또는 1회째)의
  차가 흔들림 폭 이하.
- 기준값·판정 방식은 상수·인자다 — V1 에서 답이 달라져도 함수 몸통은 바뀌지 않는다.

LLM 을 부르지 않는다. 앱 실행 경로에서 import 되지 않는다(tests/test_judge_import_rules.py).

상호작용
- 의존(import): numpy·scipy 만(앱 모듈 없음 — judge_once 결과는 속성·키로만 읽는다)
- 사용처: scripts/judge_validate.py(검증 실행기, 17-3)
"""
from __future__ import annotations

import math
import re
from statistics import mean

import numpy as np
from scipy.stats import f as f_dist

AXES = ("faithfulness", "coverage", "consistency", "readability")
AXIS_ALIASES = {"F": "faithfulness", "Cov": "coverage", "Cons": "consistency", "R": "readability"}
GATE_AXES = ("faithfulness", "consistency")  # 게이트가 걸린 축(설계 5.4) — 탐지 기준이 더 엄격하다(7.6)

ICC_THRESHOLD = 0.75  # 설계 7.6: ICC 점추정 0.75 이상
CI_LEVEL = 0.95
SUMMARY_MODES = ("mean", "first")  # 독립성·sham 에서 회차를 하나로 줄이는 방식(카-4: 1회째 / 평균)
DEFAULT_SUMMARY = "mean"  # Q11 추천
OTHER_MIN_HIT_RATE = 3 / 5  # 7.6: 필수 내용·표기 결함은 5회 중 3회 이상 탐지
OTHER_MIN_DEFECT_SHARE = 5 / 6  # 7.6: 필수 내용·표기 결함 6개 중 5개 이상(개수에 비례, 올림)
HUMAN_MARGIN = 0.05  # 7.6: 시스템↔사람 일치율이 사람↔사람보다 5%p 넘게 낮지 않음
FLOAT_TOL = 1e-9
OVERLAP_MIN = 10  # 잘린 인용(표기 문장 규칙은 120자로 자른다)과 심은 문장의 앞뒤가 이만큼 겹치면 짚은 것으로 본다

_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------- ICC(A,1)

def _mean_squares(x: np.ndarray) -> tuple[float, float, float]:
    """두 방향 분산분석의 평균 제곱(대상 MSR, 평가자 MSC, 잔차 MSE)."""
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * float(((x.mean(axis=1) - grand) ** 2).sum())
    ss_cols = n * float(((x.mean(axis=0) - grand) ** 2).sum())
    ss_total = float(((x - grand) ** 2).sum())
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    return msr, msc, max(mse, 0.0)


def icc_a1(matrix, *, level: float = CI_LEVEL) -> dict:
    """ICC(A,1) 과 신뢰구간. matrix = 대상(성적서) × 평가자(회차). 빈칸(None·NaN)이 있으면 ValueError — 호출자가 먼저 정리한다.

    돌려주는 키: icc, ci_low, ci_high, n_targets, n_raters, zero_variance, note.
    """
    x = np.asarray(matrix, dtype=float) if len(matrix) else np.zeros((0, 0))
    if x.ndim != 2:
        raise ValueError("matrix 는 대상 × 평가자 2차원이어야 한다")
    if np.isnan(x).any():
        raise ValueError("빈칸(None·NaN)이 있다 — 대상을 빼거나 채운 뒤 부른다")
    n, k = x.shape
    out = {"icc": None, "ci_low": None, "ci_high": None, "n_targets": n, "n_raters": k, "zero_variance": False, "note": ""}
    if n == 0 or k == 0:
        out["note"] = "빈 입력"
        return out
    if np.all(x == x.flat[0]):
        out["zero_variance"] = True
        out["note"] = "분산 0 — 모든 대상·모든 회차 점수가 같다(Q10 규칙: 통과로 본다)"
        return out
    if n < 2 or k < 2:
        out["note"] = "대상 2개·평가자 2명 이상이어야 계산된다"
        return out
    msr, msc, mse = _mean_squares(x)
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    icc = (msr - mse) / denom
    out["icc"] = float(icc)
    if mse <= FLOAT_TOL and msc <= FLOAT_TOL:
        out["ci_low"] = out["ci_high"] = 1.0
        out["note"] = "회차끼리 완전히 같다(잔차 0) — 신뢰구간은 1 로 적는다"
        return out
    if icc >= 1.0:
        out["ci_low"] = out["ci_high"] = 1.0
        return out
    # McGraw & Wong (1996) Case 2A, ICC(A,1) 신뢰구간
    alpha = 1.0 - level
    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    f_star = f_dist.ppf(1 - alpha / 2, n - 1, v)
    f_star2 = f_dist.ppf(1 - alpha / 2, v, n - 1)
    low = n * (msr - f_star * mse) / (f_star * (k * msc + (k * n - k - n) * mse) + n * msr)
    high = n * (f_star2 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_star2 * msr)
    out["ci_low"], out["ci_high"] = float(low), float(high)
    return out


# ---------------------------------------------------------------- judge_once 결과 읽기

def _parts(result) -> tuple[str | None, dict | None, dict]:
    """judge_once 결과(JudgeRun) 또는 그것을 적은 dict({"gate","scorecard","record"}) → (gate, scorecard, record)."""
    if result is None:
        return None, None, {}
    if isinstance(result, dict):
        return result.get("gate"), result.get("scorecard"), result.get("record") or {}
    return getattr(result, "gate", None), getattr(result, "scorecard", None), getattr(result, "record", None) or {}


def _ratio(card: dict, num: str, den: str) -> float | None:
    if not isinstance(card, dict) or num not in card or den not in card:
        return None
    if not card[den]:
        return None  # 분모 0 — 비율이 없다
    return card[num] / card[den]


def _count(card: dict, key: str) -> float | None:
    if not isinstance(card, dict) or card.get(key) is None:
        return None
    return float(card[key])


def axis_scores(judge_result) -> dict[str, float | None]:
    """축별 점수 넷(Q10). 실패 회차(scorecard 없음)·키 없음·분모 0 이면 그 축은 None.

    - faithfulness: 부합 비율 = matched / checked(검사한 숫자·이름·비교·정도 주장 중 데이터와 맞은 것)
    - coverage: 서술 비율 = described / applicable
    - consistency: 모순 건수 = contradictions
    - readability: 위반 건수 = violations(코드 규칙 2개. LLM 표시 llm_flags 는 넣지 않는다 — 고객 묶음 bundle.passed 와 같은 셈)
    """
    _, card, _ = _parts(judge_result)
    card = card if isinstance(card, dict) else {}
    return {
        "faithfulness": _ratio(card.get("faithfulness"), "matched", "checked"),
        "coverage": _ratio(card.get("coverage"), "described", "applicable"),
        "consistency": _count(card.get("consistency"), "contradictions"),
        "readability": _count(card.get("readability"), "violations"),
    }


def _scores(x) -> dict[str, float | None]:
    """이미 축 점수 dict 이면 그대로, 아니면 judge_once 결과로 보고 축 점수를 뽑는다."""
    if isinstance(x, dict) and set(x) == set(AXES):
        return x
    return axis_scores(x)


def gate_of(judge_result) -> str | None:
    return _parts(judge_result)[0]


# ---------------------------------------------------------------- 재현성

def gate_stability(gates: list[str], *, expected_runs: int | None = None) -> bool:
    """같은 성적서 반복에서 게이트 판정이 모두 같은가(설계 7.6: 5회 중 5회, 줄인 V2: 3회 중 3회).

    실패 회차(None)가 하나라도 있으면 같지 않다. expected_runs 를 주면 회차 수도 맞아야 한다.
    """
    if not gates:
        return False
    if expected_runs is not None and len(gates) != expected_runs:
        return False
    if any(g is None for g in gates):
        return False
    return len(set(gates)) == 1


def spread(values: list[float | None]) -> float | None:
    """흔들림 폭 = 최대 − 최소. 빈칸이 있으면 None(셀 수 없다)."""
    if not values or any(v is None for v in values):
        return None
    return float(max(values) - min(values))


def reproducibility(per_report_runs: dict[str, list], threshold: float = ICC_THRESHOLD, *,
                    expected_runs: int | None = None) -> dict:
    """같은 성적서를 여러 회 채점한 결과 → 축별 ICC(A,1)·통과, 성적서별 게이트 재현성·흔들림 폭.

    per_report_runs: {성적서 이름: [judge_once 결과(또는 축 점수 dict, gate 는 'gate' 키) ...]}. 회차 수는 성적서마다 같아야 한다.
    한 축에서 어느 회차가 None 이면 그 성적서는 그 축 ICC 에서 빠진다(note 에 적는다).
    """
    if not per_report_runs:
        raise ValueError("성적서가 없다")
    lengths = {len(runs) for runs in per_report_runs.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"성적서마다 회차 수가 같아야 한다: {sorted(lengths)}")
    scores = {rep: [_scores(r) for r in runs] for rep, runs in per_report_runs.items()}
    axes: dict[str, dict] = {}
    for axis in AXES:
        rows, dropped = [], []
        for rep, runs in scores.items():
            vals = [s[axis] for s in runs]
            if any(v is None for v in vals):
                dropped.append(rep)
            else:
                rows.append(vals)
        res = icc_a1(rows) if rows else icc_a1([])
        if dropped:
            res["note"] = (res["note"] + f" / 빠진 성적서(점수 없음): {', '.join(dropped)}").lstrip(" /")
        res["dropped"] = dropped
        res["passed"] = bool(not dropped and (res["zero_variance"] or (res["icc"] is not None and res["icc"] >= threshold)))
        axes[axis] = res
    gates = {}
    for rep, runs in per_report_runs.items():
        g = [gate_of(r) for r in runs]
        gates[rep] = {"gates": g, "stable": gate_stability(g, expected_runs=expected_runs)}
    spreads = {rep: {axis: spread([s[axis] for s in runs]) for axis in AXES} for rep, runs in scores.items()}
    return {
        "threshold": threshold, "n_runs": lengths.pop(), "axes": axes, "axes_passed": all(a["passed"] for a in axes.values()),
        "gates": gates, "gates_passed": all(v["stable"] for v in gates.values()), "spread": spreads,
    }


# ---------------------------------------------------------------- 결함 탐지

def _norm(text) -> str:
    return _SPACE.sub(" ", str(text or "")).strip()


def touches(quote: str, sentence: str) -> bool:
    """인용 구절이 심은 문장을 짚었나 — 한쪽이 다른 쪽에 들어 있거나, 잘린 인용의 끝과 문장 앞(또는 그 반대)이 OVERLAP_MIN 자 넘게 겹침."""
    q, s = _norm(quote), _norm(sentence)
    if not q or not s:
        return False
    if q in s or s in q:
        return True
    for size in range(min(len(q), len(s)) - 1, OVERLAP_MIN - 1, -1):
        if q.endswith(s[:size]) or s.endswith(q[:size]):
            return True
    return False


def _axis(name: str) -> str:
    axis = AXIS_ALIASES.get(name, name)
    if axis not in AXES:
        raise ValueError(f"모르는 축: {name}")
    return axis


def defect_detected(judge_result, axis: str, *, sentence: str | None = None, item_id: str | None = None) -> bool:
    """결함 하나를 이 회차가 탐지했나(Q11 추천).

    - faithfulness: 위반(violation·inference_violation) 판정 중 인용이 심은 문장(sentence)을 짚은 것이 있다
    - consistency: 모순 중 두 인용 어느 하나가 심은 문장을 짚었다
    - readability: 코드 규칙 위반(findings)이나 LLM 표시(llm_flags)의 글이 심은 문장을 짚었다
    - coverage: 지운 항목(item_id)이 "없음"(described False)으로 판정됐다
    실패 회차(해당 모듈 기록 없음)는 탐지 못 한 것으로 본다.
    """
    axis = _axis(axis)
    _, _, record = _parts(judge_result)
    if axis == "coverage":
        if not item_id:
            raise ValueError("coverage 결함은 item_id 가 필요하다")
        items = (record.get("coverage") or {}).get("items") or []
        return any(i.get("id") == item_id and i.get("described") is False for i in items)
    if not sentence or not _norm(sentence):
        raise ValueError(f"{axis} 결함은 심은 문장(sentence)이 필요하다")
    mod = record.get(axis) or {}
    if axis == "faithfulness":
        quotes = [v.get("quote") for v in mod.get("verdicts") or [] if v.get("verdict") in ("violation", "inference_violation")]
    elif axis == "consistency":
        quotes = [q for c in mod.get("contradictions") or [] for q in c.get("quotes") or []]
    else:
        quotes = [f.get("text") for f in (mod.get("findings") or []) + (mod.get("llm_flags") or [])]
    return any(touches(q, sentence) for q in quotes)


def detection_passed(by_defect: dict[str, dict], *, gate_axes=GATE_AXES, other_min_hit_rate: float = OTHER_MIN_HIT_RATE,
                     other_min_defect_share: float = OTHER_MIN_DEFECT_SHARE) -> dict:
    """결함 탐지 기준(설계 7.6). by_defect = {결함 id: {"axis": 축, "hits": [회차별 탐지 bool ...]}}.

    게이트 축 결함은 모든 회차 탐지. 그 밖의 결함은 회차의 other_min_hit_rate 이상 탐지한 결함이
    ceil(개수 × other_min_defect_share) 개 이상.
    """
    if not by_defect:
        raise ValueError("결함이 없다")
    gate_axes = {_axis(a) for a in gate_axes}
    per: dict[str, dict] = {}
    for did, d in by_defect.items():
        hits = list(d["hits"])
        if not hits:
            raise ValueError(f"{did}: 회차가 없다")
        axis = _axis(d["axis"])
        rate = sum(bool(h) for h in hits) / len(hits)
        ok = rate == 1.0 if axis in gate_axes else rate >= other_min_hit_rate - FLOAT_TOL
        per[did] = {"axis": axis, "rate": rate, "ok": ok}
    gate_ok = all(p["ok"] for p in per.values() if p["axis"] in gate_axes)
    others = [p for p in per.values() if p["axis"] not in gate_axes]
    need = math.ceil(len(others) * other_min_defect_share - FLOAT_TOL)
    others_ok = sum(p["ok"] for p in others) >= need
    return {"defects": per, "gate_axes_ok": gate_ok, "others_needed": need, "others_ok": others_ok,
            "passed": gate_ok and others_ok}


# ---------------------------------------------------------------- 독립성·sham

def summarize(values: list[float], mode: str = DEFAULT_SUMMARY) -> float:
    """회차를 하나로 줄인다 — mean(평균) 또는 first(1회째)."""
    if mode not in SUMMARY_MODES:
        raise ValueError(f"모르는 판정 방식: {mode}")
    if not values:
        raise ValueError("회차가 없다")
    return float(values[0]) if mode == "first" else float(mean(values))


def _shift(base_runs: list, other_runs: list, axes, mode: str) -> dict:
    if not base_runs or not other_runs:
        raise ValueError("바탕 회차와 비교 회차가 모두 있어야 한다")
    base = [_scores(r) for r in base_runs]
    other = [_scores(r) for r in other_runs]
    out: dict[str, dict] = {}
    for axis in axes:
        b = [s[axis] for s in base]
        o = [s[axis] for s in other]
        if any(v is None for v in b + o):
            out[axis] = {"base": None, "other": None, "diff": None, "spread": None, "ok": False, "note": "점수 없음(실패 회차 등)"}
            continue
        width = spread(b)
        diff = abs(summarize(o, mode) - summarize(b, mode))
        out[axis] = {"base": summarize(b, mode), "other": summarize(o, mode), "diff": diff, "spread": width,
                     "ok": diff <= width + FLOAT_TOL}
    return out


def independence_ok(base_runs: list, defect_runs: list, defect_axis: str, *, mode: str = DEFAULT_SUMMARY) -> dict:
    """결함을 심지 않은 축의 점수 변화가 흔들림 폭 이내인가(설계 7.6, Q11).

    base_runs: 바탕 성적서의 재현성 회차들(judge_once 결과 또는 축 점수 dict) — 흔들림 폭도 여기서 잰다.
    defect_runs: 결함본 회차들. 돌려주는 값: {"ok": bool, "axes": {축: {base, other, diff, spread, ok}}}.
    """
    defect_axis = _axis(defect_axis)
    axes = _shift(base_runs, defect_runs, [a for a in AXES if a != defect_axis], mode)
    return {"ok": all(a["ok"] for a in axes.values()), "axes": axes}


def sham_ok(base_runs: list, sham_runs: list, *, mode: str = DEFAULT_SUMMARY) -> dict:
    """뜻을 안 바꾼 수정본에서 모든 축의 점수 변화가 흔들림 폭 이내인가(설계 7.6, Q11)."""
    axes = _shift(base_runs, sham_runs, AXES, mode)
    return {"ok": all(a["ok"] for a in axes.values()), "axes": axes}


# ---------------------------------------------------------------- 사람 일치도

def agreement_rate(a: list, b: list) -> float:
    """두 판정 목록이 같은 항목의 비율. 길이가 다르거나 비면 ValueError."""
    if len(a) != len(b):
        raise ValueError("두 목록의 길이가 다르다")
    if not a:
        raise ValueError("빈 목록")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def human_agreement_ok(system_human: float, human_human: float, *, margin: float = HUMAN_MARGIN) -> bool:
    """시스템↔사람 일치율이 사람↔사람 일치율보다 margin 넘게 낮지 않은가(설계 7.6)."""
    return system_human >= human_human - margin - FLOAT_TOL
