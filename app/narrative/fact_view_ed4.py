"""app/narrative/fact_view_ed4.py — 판 4 요청의 사실 시트 보기(판 4 라운드 S2 ④, 계약 fact_sheet_contract_ed4.json).

판 3 보기(app.narrative.fact_view)는 한 글자도 바꾸지 않고, 판 4 요청(report_edition 4)만 이 갈래로 온다(fact_view.build_fact_view
가 판으로 가른다). 키 이름은 판 3 보기와 같게 두고(지시문 덜어 내기 fact_view.TRIM_STEPS 가 같은 키로 찾는다) 값의 표기와
판 4 에서 뜻이 바뀐 칸만 다르다.

- 숫자는 판 4 표기다(app.narrative.wording_ed4 — 두 저장소 공통 대조 표). 판정과 이어진 값은 프론트가 보낸 *_text 를 그대로
  싣고(value_text·margin_text·ci_*_text·f1_threshold_gain 의 *_text·display·share_text·gap_text·difference_text·
  evaluation_minus_training_text·weakest_class.f1_text), 판정과 무관한 값은 셋째 자리 끝 0 유지(plain), 비율은 백분율 첫째
  자리(37.4%)로 싣는다. 판 3 의 넷째 자리 표기(0.9767)는 싣지 않는다. S2·S3 검토 결정(공통 대조 표에 줄이 있다):
  목표값은 ed4_target(넷째 자리 이상 목표는 받은 글자 그대로 — D1), 판정과 무관한 값이 판정 대상 지표와 같은 원값이면 그
  value_text(D2), 최약 클래스 F1 은 셋째 자리로 다른 클래스와 같아 보이면 늘린 표기(D3), 지연시간은 둘째 자리(D4).
- 판정 여유: near_target 은 판 4 요청에서 '판정 여유 작음'이다(프론트 judgmentMargin 한 곳 — 구간이 목표를 포함하거나, 구간이
  없으면 3% 엄격 미만). 여기서 다시 계산하지 않는다(계획 9절). 고정 문구 18 은 판 4 요청에 없다 — 이 보기는 18 을 보지 않는다.
- 통과율(score)은 싣지 않는다(판 4 는 충족 수만 — 부록 A11).
- 참조 범위 비교는 수치 위치만(참조 범위보다 높음/안/낮음 — 우수·미흡 같은 평가 말 없음, 부록 A10). 모델 값은 KPI 원값의 판 4
  표기(required_items.benchmark_value_text_ed4), 출처 글은 기준표의 판 4 글(baselines).
- 문제 목록의 절은 판 4 자리(sections.place — 8.2 → 5.1, 9.1 → 6.1), 근거 숫자는 판 4 표기.
- 평가 − 학습 분포 차이는 판 4 전용 칸(evaluation_minus_training_text)을 %p 로 싣는다(4.5 차이 열과 같은 값).

숫자는 새로 판정하지 않는다(계획 원칙 1). 바꾸는 것은 이름과 표기뿐이다.

상호작용
- 의존(import): app.narrative.fact_view(판 3 보기의 이름표·도구), app.narrative.wording_ed4, app.narrative.required_items,
  app.narrative.sections(place)
- 사용처: app.narrative.fact_view.build_fact_view(판 4 갈래), tests
"""
from __future__ import annotations

from typing import Any

from app.narrative import fact_view as v3
from app.narrative.required_items import benchmark_value_text_ed4, weakest_ties
from app.narrative.schemas import FactSheet
from app.narrative.sections import place
from app.narrative.wording_ed4 import (
    POSITION_KO,
    ed4_ci_bound,
    ed4_distinct,
    ed4_given,
    ed4_judged,
    ed4_latency,
    ed4_percent,
    ed4_percent_point,
    ed4_plain,
    ed4_range_text,
    ed4_ratio_percent,
    ed4_signed_judged,
    ed4_target,
    imbalance_texts_ed4,
    judged_value_text_ed4,
    metric_value_ed4,
    same_metric_value_ed4,
    same_value_text_ed4,
    weakest_f1_text_ed4,
)

EDITION = 4

# 4.3 판정 열의 말(고정 문구 19·부록 A5 — '정보 제공' 대신 '참고 지표').
STATUS_KO = {"pass": "충족", "fail": "미달", "warning": "경고", "info": "참고 지표"}
# 판정 여유(고정 문구 19 — 4.3 판정 여유 열의 값과 표 아래 문장의 뜻). 칸 이름에는 숫자를 두지 않는다 — 숫자 허용 목록이 보기의
# 값에서 숫자를 뽑으므로, 칸 이름에 95·3 을 적으면 그 갈래가 없는 성적서에서도 지어낸 '3건'·'95%' 를 받는 구멍이 된다(S2·S3 검토
# D5). '작음'의 뜻은 판정 칸의 MARGIN_MEANING_KEY 에 그 성적서에 해당하는 갈래만 숫자와 함께 싣는다.
MARGIN_MEANING_KEY = "판정 여유가 작음의 뜻"
MARGIN_KEY = f"판정 여유(작음·충분 — 뜻은 판정의 '{MARGIN_MEANING_KEY}')"
_MARGIN_BY_CI = "95% 신뢰구간이 목표값을 포함하면 작음"
_MARGIN_BY_SHARE = "신뢰구간이 없는 지표는 목표값과의 차이가 목표값의 3% 미만이면 작음"
MARGIN_KO = {True: "작음", False: "충분"}
NO_TARGET = "없음(참고 지표)"


def _num(value: Any) -> Any:
    """판정과 무관한 실수는 판 4 plain(셋째 자리 끝 0 유지 — 글자). 정수·글·None 은 그대로."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    return ed4_plain(value)


def _metric_value(m) -> str:
    """지표 값의 판 4 표기 — wording_ed4.metric_value_ed4(규칙 문장·숫자 허용 목록과 같은 한 곳)."""
    return metric_value_ed4(m)


def _same_metric_value(fs: FactSheet, metric_id: str, value: float | None) -> str | None:
    """파생값에 실린 값이 판정 대상 지표의 값이면 그 지표의 표기(wording_ed4.same_metric_value_ed4)."""
    return same_metric_value_ed4(fs.metrics, metric_id, value)


def margin_meaning(fs: FactSheet) -> str | None:
    """판정 여유 '작음'의 뜻 — 이 성적서의 판정 대상 지표에 해당하는 갈래만(구간이 있는 지표 → 95% 신뢰구간, 구간이 없는 지표 →
    목표값의 3%). 판정은 프론트(near_target)가 했고 여기서는 뜻만 적는다. 판정 대상이 없으면 None."""
    judged = [m for m in fs.metrics if m.threshold is not None]
    parts = []
    if any(m.ci_contains_target is not None for m in judged):
        parts.append(_MARGIN_BY_CI)
    if any(m.ci_contains_target is None for m in judged):
        parts.append(_MARGIN_BY_SHARE)
    return ", ".join(parts) or None


def _verdict(fs: FactSheet, derived: dict) -> dict:
    out = v3._verdict(fs, derived)
    out.pop("통과율(참고, %)", None)  # 판 4 는 충족 수만(A11)
    out[MARGIN_MEANING_KEY] = margin_meaning(fs)
    return out


def _metrics(fs: FactSheet, task_type: str) -> list[dict]:
    rows = []
    for m in fs.metrics:
        if m.metric_id in v3.TABLE_METRICS:
            continue
        judged = m.threshold is not None
        row: dict[str, Any] = {
            "지표": m.display_name,
            "값": _metric_value(m),
            "목표값": ed4_target(m.threshold) if judged else NO_TARGET,
            "결과": STATUS_KO.get(m.status, m.status),
            "핵심 지표": "예" if m.is_core else "아니오",
            "좋은 방향": None if m.higher_is_better is None else ("높을수록 좋음" if m.higher_is_better else "낮을수록 좋음"),
            "β": m.beta,
            "평균 방식": v3.averaging_text(m.metric_id, task_type),
        }
        if m.margin is not None:
            row["목표와의 차이(0 이상이면 충족 쪽)"] = m.margin_text or ed4_judged(m.margin, 0, True, m.status == "pass")
        if judged:
            row[MARGIN_KEY] = MARGIN_KO[bool(m.near_target)]
        if m.ci_low is not None and m.ci_high is not None:
            low = m.ci_low_text or (ed4_ci_bound(m.ci_low, m.threshold) if judged else ed4_plain(m.ci_low))
            high = m.ci_high_text or (ed4_ci_bound(m.ci_high, m.threshold) if judged else ed4_plain(m.ci_high))
            row["95% 신뢰구간"] = f"{low}~{high}"
            if m.ci_contains_target is not None:
                row["구간이 목표값을 포함"] = v3._yes(m.ci_contains_target)
        rows.append(row)
    return rows


def _confusion(fs: FactSheet, derived: dict, task_type: str) -> dict | None:
    out = v3._confusion(fs, derived, task_type)
    conf = derived.get("confusion")
    if out is None or not conf or "합계" not in out:
        return out
    total = conf.get("total") or 0
    totals = out["합계"]
    for old in ("맞음 비율(%)", "틀림 비율(%)", "TP 비율(%)", "FP 비율(%)", "FN 비율(%)", "TN 비율(%)"):
        totals.pop(old, None)
    if total:
        totals["맞음 비율"] = ed4_ratio_percent(conf["correct"], total)
        totals["틀림 비율"] = ed4_ratio_percent(conf["misclassified"], total)
        for key in ("tp", "fp", "fn", "tn"):
            if key in conf:
                totals[f"{key.upper()} 비율"] = ed4_ratio_percent(conf[key], total)
    return out


def _same(fs: FactSheet, value: Any) -> Any:
    """판정과 무관한 자리의 값 — 같은 원값의 판정 대상 지표가 있으면 그 value_text(D2), 아니면 plain. 정수·글·None 은 그대로."""
    return same_value_text_ed4(fs.metrics, value)


def _class_f1(fs: FactSheet, label: str, value: Any) -> Any:
    """클래스 하나의 F1 — 최약 클래스면 최약 클래스 F1 글자(프론트 facts.weakest_class.f1_text — 같은 수는 같은 글자), 그 밖은
    같은 원값의 판정 대상 지표 글자(D2)나, 셋째 자리로 최약 클래스와 같아 보이면 서로 달라질 때까지 늘린 표기(D3)."""
    w = fs.facts.weakest_class if fs.facts is not None else None
    if w is None or value is None or isinstance(value, bool):
        return _same(fs, value)
    if label == w.label:
        return weakest_f1_text_ed4(fs)
    return judged_value_text_ed4(fs.metrics, value) or ed4_distinct(value, [w.f1])


def _per_class(fs: FactSheet) -> dict | None:
    out: dict[str, Any] = {}
    if fs.per_class:
        out["클래스별"] = [
            {
                "클래스": p.label, "Precision": _same(fs, p.precision), "Recall": _same(fs, p.recall),
                "F1": _class_f1(fs, p.label, p.f1), "표본 수": p.support,
            }
            for p in fs.per_class
        ]
    if fs.class_averages:
        out["평균 줄"] = [
            {"평균": a.label, "Precision": _same(fs, a.precision), "Recall": _same(fs, a.recall), "F1": _same(fs, a.f1), "표본 수": a.support}
            for a in fs.class_averages
        ]
    return out or None


def _imbalance_texts(ie) -> dict[str, str | None]:
    """불균형 영향 값의 판 4 표기(wording_ed4.imbalance_texts_ed4 — 규칙 문장·숫자 허용 목록과 같은 한 곳)."""
    return imbalance_texts_ed4(ie)


def _distribution(fs: FactSheet, derived: dict, task_type: str) -> dict | None:
    out: dict[str, Any] = {}
    dist = derived.get("distribution")
    if fs.distribution and fs.distribution.class_distribution:
        out["클래스별 건수"] = fs.distribution.class_distribution
        if dist:
            total = dist.get("total") or 0
            out["백분율(%, 분모는 평가 표본 수)"] = {
                k: ed4_ratio_percent(n, total) for k, n in fs.distribution.class_distribution.items()
            }
    ie = fs.imbalance_effect
    texts = _imbalance_texts(ie) if ie is not None else None
    if fs.distribution and fs.distribution.imbalance_ratio is not None:
        # 1.5 와 비교되는 값 — 판 4 표기(프론트 display, 없으면 M23 원값의 plain).
        out["불균형 비율"] = texts["imbalance_ratio"] if texts else ed4_plain(fs.distribution.imbalance_ratio)
    if task_type == "multilabel" and out:
        out["유의"] = "레이블 등장 횟수이다. 한 표본에 레이블이 여럿일 수 있어 백분율을 모두 더하면 전체를 넘을 수 있다."
    if ie is not None:
        noun = "레이블" if ie.accuracy_minus_macro_f1 is None else "클래스"
        out["불균형 영향(평가 엔진 판단)"] = {
            "판단": v3.IMBALANCE_KO[ie.level],
            "불균형 비율": texts["imbalance_ratio"],
            "정확도 − 매크로 F1": texts["accuracy_minus_macro_f1"],
            f"다수 − 소수 {noun} F1 차이": texts["majority_minus_minority_f1"],
            f"다수 {noun}": ie.majority_class,
            f"소수 {noun}": ie.minority_class,
        }
    return out or None


def _error_ratios(fs: FactSheet) -> dict | None:
    er = fs.error_ratios
    if er is None:
        return None

    def part(p):
        return None if p is None else {"건수": p.count, "비율": ed4_ratio_percent(p.count, er.misclassified)}

    return {
        "단위": "틀린 레이블 칸" if er.unit == "label_cell" else "오분류 표본",
        "확률이 있는 오분류 수": er.misclassified,
        "경계 오류(결정 경계와의 거리가 기준 미만)": part(er.boundary)
        or ("계산하지 않음 — 예측값을 직접 제출해 결정 임계값을 모름" if er.boundary_note == "threshold_unknown" else "계산하지 않음"),
        "확신 오류(틀린 쪽 확률이 기준 이상)": part(er.confident),
        "경계 거리 기준": er.boundary_margin,
        "확신 기준": er.confident_level,
    }


_RISK_SIDE_KO = {"fn": "FN(놓침)", "fp": "FP(잘못 잡음)"}


def _threshold_gain(fs: FactSheet, derived: dict) -> dict | None:
    g = fs.f1_threshold_gain
    if g is None:
        return None
    current = fs.derived_prediction.threshold if fs.derived_prediction else None
    out: dict[str, Any] = {
        # 의뢰자가 준 결정 임계값은 늘이지도 채우지도 않는다(0.5 — '0.500' 아님).
        "현재 결정 임계값": ed4_given(current) if isinstance(current, (int, float)) and not isinstance(current, bool) else None,
        "현재 F1": g.current_f1_text or _same_metric_value(fs, "M4", g.current_f1),
        "F1 이 가장 높은 결정 임계값": g.best_threshold_text or ed4_given(g.best_threshold),
        "그 임계값의 F1": g.max_f1_text or ed4_plain(g.max_f1),
        "현재 Recall": g.current_recall_text,
        "그 임계값의 Recall": g.best_recall_text,
        "현재 Precision": g.current_precision_text,
        "그 임계값의 Precision": g.best_precision_text,
        "향상폭(이 데이터에서 오를 수 있는 최대폭)": g.gain_text or ed4_judged(
            g.gain, 0.1, True, bool(derived.get("threshold_tuning")),
        ),
        "평가 엔진 판단": "현재 임계값이 부적합(향상폭 0.1 이상)" if derived.get("threshold_tuning") else "조정 여지가 크지 않음(향상폭 0.1 미만)",
        "판단 기준": "F1 만 본 판단(Recall·Precision 등 다른 지표에 대한 판단이 아님)",
    }
    rs = g.risk_side
    if rs is not None:
        out["의뢰자가 더 위험하다고 밝힌 오류 쪽 지표"] = {
            "오류": _RISK_SIDE_KO[rs.side],
            "지표": rs.display_name,
            "현재 목표 충족": v3._yes(rs.meets_target_now),
            "F1 이 가장 높은 결정 임계값에서 목표 충족": v3._yes(rs.meets_target_at_best),
        }
    return out


def _latency(fs: FactSheet) -> dict | str:
    """지연시간 — 4.8 표와 같은 둘째 자리 글자(D4 — wording_ed4.ed4_latency). 측정하지 않았으면 판 3 과 같은 글."""
    out = v3._latency(fs)
    if not isinstance(out, dict):
        return out
    return {k: ed4_latency(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for k, v in out.items()}


def _facts(fs: FactSheet, task_type: str) -> dict | None:
    facts = fs.facts
    if facts is None:
        return None
    noun = "레이블" if task_type == "multilabel" else "클래스"
    out: dict[str, Any] = {}
    w = facts.weakest_class
    if w is not None:
        out[f"F1 이 가장 낮은 {noun}"] = {
            noun: w.label,
            "F1": weakest_f1_text_ed4(fs),
            "표본 수": w.support,
            "비율": ed4_percent(w.share_text),
            "매크로 F1": _same(fs, w.macro_f1),
            "매크로 F1 과의 차이": w.gap_text,
            "표본이 작음(30건 미만 또는 5% 미만)": v3._yes(w.low_support),
            "매크로 F1 보다 0.1 이상 낮음": v3._yes(w.far_below_macro),
            f"F1 이 같은 {noun}": weakest_ties(fs, edition=EDITION)[0] or None,
        }
    if facts.low_support_classes:
        out[f"표본이 작은 {noun}"] = [
            {noun: r.label, "표본 수": r.support, "비율": ed4_percent(r.share_text)} for r in facts.low_support_classes
        ]
    if facts.top_confusion_pairs:
        out["가장 많이 헷갈린 쌍(실제 → 예측)"] = [
            {"실제": p.actual, "예측": p.predicted, "건수": p.count, "전체 오분류 중 비율": ed4_percent(ed4_plain(p.share_of_errors))}
            for p in facts.top_confusion_pairs
        ]
    bias = facts.prediction_bias
    if bias is not None and bias.rows:
        out["예측 편향"] = {
            "무엇을 비교": v3.BIAS_UNIT_KO[bias.unit],
            "줄": [
                {noun: r.label, "예측 수": r.predicted, "실제 수": r.actual, "차이(예측 − 실제)": r.difference}
                for r in bias.rows
            ],
        }
    dr = facts.declared_risk_errors
    if dr is not None:
        share = dr.share_text or (ed4_plain(dr.share) if dr.share is not None else None)
        out["의뢰자가 더 위험하다고 밝힌 오류"] = {
            "오류": _RISK_SIDE_KO[dr.side],
            "건수": dr.count,
            "전체 오류 수": dr.total_errors,
            "전체 오류 중 비율": ed4_percent(share) if share is not None else None,
        }
    g = facts.precision_recall_gap
    if g is not None:
        out["양성 클래스 Precision 과 Recall"] = {
            "Precision": _same_metric_value(fs, "M2", g.precision),
            "Recall": _same_metric_value(fs, "M3", g.recall),
            "차이(Precision − Recall)": g.difference_text,
            "치우침(|차이| 0.1 이상)": v3._yes(g.flagged),
        }
    ag = facts.average_gap
    if ag is not None:
        other = "가중 평균" if ag.kind == "macro_vs_weighted" else "마이크로 평균"
        out[f"매크로 평균과 {other}의 F1 차이"] = {
            "매크로 F1": _same_metric_value(fs, "M4", ag.macro_f1),
            f"{other} F1": _same(fs, ag.other_f1),
            "차이": ag.difference_text,
            "기준 클래스" if ag.kind == "macro_vs_weighted" else "F1 이 가장 낮은 레이블": {
                noun: ag.reference_class.label,
                "표본 수": ag.reference_class.support,
                "F1": _class_f1(fs, ag.reference_class.label, ag.reference_class.f1),
            },
        }
    da = facts.declared_average
    if da is not None:
        out["의뢰자 선언에 맞춘 평균 방식의 결과"] = {
            "선언": v3.PRIORITY_DECL_KO[da.class_priority],
            "평균 방식": "가중 평균" if da.average == "weighted" else "마이크로 평균",
            "Precision": _same(fs, da.precision),
            "Recall": _same(fs, da.recall),
            "F1": _same(fs, da.f1),
        }
    out["학습↔평가 분포 비교"] = v3.SHIFT_STATUS_KO[facts.distribution_shift_status]
    if facts.distribution_shift:
        out["학습↔평가 분포 차이"] = [
            {
                noun: r.label,
                "학습 비율": ed4_percent(ed4_plain(r.training_ratio)),
                "평가 비율": ed4_percent(ed4_plain(r.evaluation_ratio)),
                # 판 4 전용 칸(평가 − 학습 — 4.5 차이 열과 같은 값). 없으면(형식 오류로 버려진 경우) 같은 규칙으로 찍는다.
                "차이(평가 − 학습)": ed4_percent_point(
                    r.evaluation_minus_training_text
                    or ed4_signed_judged(r.evaluation_ratio - r.training_ratio, 0.1, r.flagged)
                ),
                "10%p 이상": v3._yes(r.flagged),
            }
            for r in facts.distribution_shift
        ]
    if facts.ci_skipped:
        names, protect = v3.metric_names(fs), v3.class_names(fs)
        out["신뢰구간을 산출하지 않은 지표"] = [
            {"지표": s.display_name, "사유": v3.name_metric_ids(s.reason, names, protect) if s.reason else "제공되지 않음"}
            for s in facts.ci_skipped
        ]
    decl = facts.declarations
    out["의뢰자 선언"] = {
        "위험한 오류 방향": v3.RISK_KO.get(decl.risk_direction) if decl.risk_direction else None,
        "클래스 중요도": v3.PRIORITY_DECL_KO.get(decl.class_priority) if decl.class_priority else None,
        "사용 방식": v3.USAGE_KO.get(decl.usage_mode) if decl.usage_mode else None,
    }
    return out


def _benchmarks(benchmark_refs: list[dict], fs: FactSheet) -> dict | None:
    if not benchmark_refs:
        return None
    return {
        "기준의 출처·성격": benchmark_refs[0].get("source_note"),
        "지표별": [
            {
                "지표": r["metric"],
                "모델 값": benchmark_value_text_ed4(fs.metrics, r),
                "참조 범위": ed4_range_text(r["ref_low"], r["ref_high"]),
                # 수치 위치만(낮을수록 좋은 지표도 수치로 — Hamming Loss 0.03 은 '참조 범위보다 낮음'). 평가 말을 싣지 않는다.
                "위치": POSITION_KO[r["position"]],
                "목표값": ed4_target(r["target_value"]) if r.get("target_value") is not None else None,
                "목표값의 위치": v3.TARGET_POSITION_KO.get(r.get("target_position")) if r.get("target_position") else None,
            }
            for r in benchmark_refs
        ],
    }


# ── 문제 근거(evidence)의 판 4 표기 ────────────────────────────────────────────────
# 근거는 프론트 problemList 가 종류마다 만드는 자유 dict 다. 판정과 이어진 숫자는 같은 사실 시트의 판 4 표기 칸으로 바꾸고
# (지표 값·목표·마진, 최약 클래스 비율·차이, 분포 차, 임계값 조정의 *_text), 남은 실수는 plain 으로 찍는다. 판 4 표기로
# 옮긴 뒤 남는 엔진·프론트 보조 키(*_raw·*_text·risk_side)는 영어 필드명이라 싣지 않는다.

_METRIC_KINDS = frozenset({"core_fail", "noncore_fail", "core_near_target"})
_GAIN_KEYS = ("max_f1", "current_f1", "gain", "best_threshold", "notable")


def _metric_entry(m, entry: dict) -> dict:
    out = dict(entry)
    if "value" in out:
        out["value"] = _metric_value(m)
    if "target" in out and m.threshold is not None:
        out["target"] = ed4_target(m.threshold)
    if "margin" in out and m.margin is not None:
        out["margin"] = m.margin_text or ed4_judged(m.margin, 0, True, m.status == "pass")
    return out


def _evidence_ed4(problem, fs: FactSheet) -> dict:
    ev = problem.evidence
    kind = problem.kind
    if kind in _METRIC_KINDS:
        by_name = {m.display_name: m for m in fs.metrics}
        return {
            k: _metric_entry(by_name[k], e) if isinstance(e, dict) and k in by_name else e for k, e in ev.items()
        }
    facts = fs.facts
    if kind == "weakest_class" and facts is not None and facts.weakest_class is not None:
        w = facts.weakest_class
        out = {}
        for label, e in ev.items():
            if isinstance(e, dict) and label == w.label:
                e = {**e, "share": ed4_percent(w.share_text)}
                if "gap_from_macro" in e and w.gap_text is not None:
                    e["gap_from_macro"] = w.gap_text
                if "f1" in e:
                    e["f1"] = weakest_f1_text_ed4(fs)
                if e.get("macro_f1") is not None:
                    e["macro_f1"] = _same(fs, w.macro_f1)
            out[label] = e
        return out
    if kind == "distribution_shift" and facts is not None:
        rows = {r.label: r for r in facts.distribution_shift}
        out = {}
        for label, e in ev.items():
            r = rows.get(label)
            if isinstance(e, dict) and r is not None:
                e = {
                    **e,
                    "training_ratio": ed4_percent(ed4_plain(r.training_ratio)),
                    "evaluation_ratio": ed4_percent(ed4_plain(r.evaluation_ratio)),
                    "difference": r.difference_text,
                }
            out[label] = e
        return out
    if kind == "imbalance_effect" and fs.imbalance_effect is not None:
        # 프론트는 근거에 판 3 파서 결과(display 도 판 3 표기 — 52.9231·0.1333)를 펼쳐 싣는다. 같은 값의 판 4 표기는 사실 시트
        # imbalance_effect(판 4 요청은 판 4 display) 한 곳에서 — 근거의 display 를 그 표기로 바꿔 끼운다(규칙 문장·허용 목록과 같은 글자).
        texts = imbalance_texts_ed4(fs.imbalance_effect)
        return {**ev, "display": {k: t for k, t in texts.items() if k in ev and t is not None}}
    if kind == "threshold_tuning" and fs.f1_threshold_gain is not None:
        g = fs.f1_threshold_gain
        shown = {
            "max_f1": g.max_f1_text or ed4_plain(g.max_f1),
            "current_f1": g.current_f1_text or ed4_plain(g.current_f1),
            "gain": g.gain_text or ed4_plain(g.gain),
            "best_threshold": g.best_threshold_text or ed4_given(g.best_threshold),
            "notable": g.notable,
        }
        return {k: shown[k] for k in _GAIN_KEYS if k in ev and shown[k] is not None}
    return ev


def _drop_helper_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _drop_helper_keys(val) for k, val in obj.items()
            if not (isinstance(k, str) and (k.endswith("_raw") or k.endswith("_text") or k == "risk_side"))
        }
    if isinstance(obj, list):
        return [_drop_helper_keys(x) for x in obj]
    return obj


def _plain_floats(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain_floats(val) for k, val in obj.items()}
    if isinstance(obj, list):
        return [_plain_floats(x) for x in obj]
    return _num(obj)


def translate_evidence_ed4(problem, fs: FactSheet) -> Any:
    ev = _evidence_ed4(problem, fs)
    if problem.kind == "imbalance_effect":
        # 불균형 근거는 엔진 값 펼침 — display(판 4 요청은 판 4 표기)가 숫자 칸을 덮는다(판 3 translate_evidence 와 같은 방식).
        ev = v3.translate_evidence(ev)
        return _plain_floats(_drop_helper_keys(ev))
    return _plain_floats(v3.translate_evidence(_drop_helper_keys(ev)))


def _problems(fs: FactSheet) -> list[dict]:
    return [
        {
            "번호": p.id,
            "우선순위": v3.PRIORITY_KO[p.priority],
            "문제": p.title,
            # 판 4 절 번호(칸 이름 → 자리). '절'을 붙인다 — 서술이 옮겨 적어도 절 번호로 읽힌다.
            "드러난 절": f"{place(p.source_section, edition=EDITION)}절" if p.source_section else None,
            "권고 절": f"{place(p.recommend_section, edition=EDITION)}절" if p.recommend_section else None,
            "근거": translate_evidence_ed4(p, fs) or None,
            "덧붙임": p.note,
        }
        for p in fs.problems
    ]


def build_fact_view_ed4(
    fs: FactSheet, derived: dict, benchmark_refs: list[dict], *, task_type: str, task_label: str, report_purpose: str,
) -> dict:
    """판 4 요청의 사실 보기(한국어 이름, 판 4 표기). 키 차례·이름은 판 3 보기와 같다. None·빈 칸은 뺀다."""
    view = {
        "평가 개요": v3._overview(fs, task_label, report_purpose),
        "판정": _verdict(fs, derived),
        "지표": _metrics(fs, task_type),
        "혼동행렬": _confusion(fs, derived, task_type),
        "클래스별 성능": _per_class(fs),
        "분포·불균형": _distribution(fs, derived, task_type),
        "오분류의 성격": _error_ratios(fs),
        "임계값 조정 여지": _threshold_gain(fs, derived),
        "예측 파생": v3._derivation(fs),
        "지연시간": _latency(fs),
        "신뢰구간 방식": (
            v3.name_metric_ids(fs.ci_method_note, v3.metric_names(fs), v3.class_names(fs)) if fs.ci_method_note else None
        ),
        "서술용 파생값": _facts(fs, task_type),
        "벤치마크(내부 참조 기준)": _benchmarks(benchmark_refs, fs),
        "문제 목록": _problems(fs),
        "덜어 낸 표(일부만 실림)": v3.omitted_counts(fs),
    }
    return v3._clean(view)
