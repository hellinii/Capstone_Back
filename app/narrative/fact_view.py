"""app/narrative/fact_view.py — AI 가 읽는 사실 시트 보기(한국어 이름) (서술 라운드 S5, 점검 재-4·생-8).

옛 지시문은 사실 시트를 영어 키 그대로(JSON) 보냈다. 그러면 AI 가 'best_threshold'·'low_support'·'FAIL' 같은 필드명과
코드값을 문장에 베낀다 — 운영 서술에 "판정은 FAIL이다"가 실제로 나왔다(라운드 기록 장부 15). 작성 규칙(서술 설계 D37)은
영어 코드값과 필드명을 문장에 쓰지 말라고 한다. 그래서 작업 경로의 지시문은 이 보기를 싣는다:

- 키는 한국어 이름, 코드값(판정·상태·선언·단위·문제 종류·우선순위)은 한국어 말로 바꾼다.
- 지표 번호(M1…)는 싣지 않는다 — 지표는 이름으로만 부른다(고정 문구 18 의 'Accuracy(M1)' 표기는 지시문에서 뺀다).
- 판정과 비교되는 값은 판정과 같은 쪽으로 읽히게 만든 표기(value_text·margin_text·*_text·display·gain_text)를 그대로
  싣는다. 표기가 없는 값만 숫자로 싣는다 — 판정과 무관한 값은 소수 넷째 자리로 줄인다(숫자 허용 목록은 원본 사실
  시트로 만들고, 넷째 자리 반올림 표기를 포함한다 — app.narrative.grounding._add).
- 고객 입력은 아는 묶음만 한국어 이름으로 싣는다. 모르는 묶음은 싣지 않는다(서술 설계 D34 — 회사·과제 정보가 섞여
  와도 외부 API 로 나가지 않게).

숫자는 여기서 새로 계산하지 않는다(계획 원칙 1). 바꾸는 것은 이름과 표기뿐이다.

상호작용
- 의존(import): app.narrative.schemas(FactSheet), app.narrative.fallback(판정 이름 한 벌), app.narrative.required_items(판정 불가
  원인 이름). 분류 유형 이름은 호출부(app.narrative.prompt 의 TASK_TYPE_LABELS)가 넘긴다 — 순환 import 를 피한다.
- 사용처: app.narrative.prompt(장별 지시문), tests
"""
from __future__ import annotations

import copy
import re
from typing import Any

from app.narrative.fallback import _VERDICT_LABEL as VERDICT_KO  # 성적서 8.2 판정 상자와 같은 이름 한 벌
from app.narrative.required_items import _CAUSE as CAUSE_KO
from app.narrative.schemas import FactSheet

# 성적서 용도 이름(프론트 reportConstants.REPORT_PURPOSE_LABEL) — facts.report_purpose_label 이 없는 옛 요청에만 쓴다.
REPORT_PURPOSE_KO = {"internal": "내부 검증", "external": "외부 제출", "project": "과제 산출물"}
STATUS_KO = {"pass": "충족", "fail": "미달", "warning": "경고", "info": "목표값 없음(참고)"}
PRIORITY_KO = {"HIGH": "높음", "MEDIUM": "중간", "LOW": "낮음"}
IMBALANCE_KO = {
    "affected": "불균형 영향 있음(불균형 비율 1.5 이상이고 차이가 0.1 이상)",
    "no_distortion": "불균형은 있으나 성능 왜곡은 관찰되지 않음(불균형 비율 1.5 이상, 차이는 0.1 미만)",
    "mild": "불균형이 경미함(불균형 비율 1.5 미만)",
}
RISK_KO = {"fn": "FN(놓침)이 더 위험하다", "fp": "FP(잘못 잡음)이 더 위험하다", "similar": "두 오류의 위험이 비슷하다"}
PRIORITY_DECL_KO = {
    "all_equal": "모든 클래스가 동등하다",
    "larger_first": "다수 클래스를 우선한다",
    "samples_first": "샘플 단위로 중요하다",
}
USAGE_KO = {"realtime": "대화형(실시간)", "batch": "배치"}
QUALITY_KO = {"better": "우수", "within": "기준 범위 내", "worse": "미흡"}
TARGET_POSITION_KO = {"below": "참조 범위보다 낮음", "within": "참조 범위 안", "above": "참조 범위보다 높음"}
DERIVATION_KO = {
    "threshold": "확률에 결정 임계값을 적용해 예측을 만듦",
    "threshold_per_label": "레이블마다 점수에 결정 임계값을 적용해 예측을 만듦",
    "argmax": "클래스별 확률 중 가장 큰 클래스를 예측으로 고름(결정 임계값 없음)",
}
BIAS_UNIT_KO = {
    "positive_class": "양성 클래스로 예측한 수와 실제 양성 수",
    "per_class": "클래스마다 예측한 수와 실제 수",
    "label_total": "예측한 레이블 수 합계와 실제 레이블 수 합계",
}
SHIFT_STATUS_KO = {
    "compared": "비교함",
    "not_provided": "학습 분포를 받지 않아 비교하지 않음",
    "uncovered": "학습 분포가 평가 클래스를 모두 덮지 않아 비교하지 않음",
}
# 프론트가 표를 자른 칸(facts.omitted)·지시문에서 덜어 낸 칸(trim)의 한국어 이름.
OMITTED_KO = {
    "per_class": "클래스별 성능",
    "confusion": "혼동행렬",
    "confusion_counts": "클래스별 TP·FP·FN·TN",
    "class_distribution": "클래스 분포",
    "low_support_classes": "표본이 작은 클래스",
    "prediction_bias": "예측 편향",
    "distribution_shift": "학습↔평가 분포 차이",
    "facts_class_lists": "서술용 클래스 목록(표본이 작은 클래스·예측 편향·분포 차이)",
    "fixed_sentences": "성적서 고정 문장",
}
# 고객 입력의 묶음(프론트 reportMaterials.buildClientInputs 의 키). 안쪽 이름은 이미 한국어 줄 이름이다.
CLIENT_GROUPS_KO = {
    "model": "모델",
    "report_purpose_note": "성적서 용도 메모",
    "declarations": "의뢰자 선언",
    "metric_rationale": "지표 선택 이유",
    "core_metrics": "의뢰자가 지정한 핵심 지표",
    "training_data": "학습 데이터",
    "test_data": "평가 데이터",
    "ground_truth": "정답 라벨",
    "environment": "실행 환경",
    "class_label_descriptions": "클래스 설명",
}
# 문제 근거(evidence — 프론트 problemList 가 문제 종류마다 만드는 자유 dict)의 키.
EVIDENCE_KEYS_KO = {
    "value": "값",
    "target": "목표값",
    "margin": "목표와의 차이",
    "cause": "원인",
    "metrics": "지표",
    "judged_count": "판정 대상 지표 수",
    "f1": "F1",
    "support": "표본 수",
    "share": "비율",
    "macro_f1": "매크로 F1",
    "gap_from_macro": "매크로 F1 과의 차이",
    "low_support": "표본이 작음",
    "training_ratio": "학습 비율",
    "evaluation_ratio": "평가 비율",
    "difference": "차이",
    "max_f1": "곡선 위 최대 F1",
    "current_f1": "현재 F1",
    "gain": "향상폭",
    "best_threshold": "최대 F1 의 결정 임계값",
    "notable": "임계값 부적합",
    "imbalance_ratio": "불균형 비율",
    "accuracy_minus_macro_f1": "정확도 − 매크로 F1",
    "majority_minus_minority_f1": "다수 − 소수 F1 차이",
    "majority_class": "다수 클래스",
    "minority_class": "소수 클래스",
    "level": "판단",
}
# 근거에서 싣지 않는 키 — 같은 뜻의 한국어 칸이 따로 있다(reason ↔ cause). 표시 문자열(display·gain_text)은 버리지 않고
# 같은 이름의 숫자 칸 값을 그 표기로 바꾼 뒤 뺀다 — 엔진은 숫자 칸을 넷째 자리로 반올림하고 판정과 같은 쪽으로 읽히는 표기는
# display 에 따로 두므로, 숫자 칸만 실으면 0.09996 이 0.1 로 보인다(S5 검토 prompt-1).
EVIDENCE_SKIP = frozenset({"reason"})
# 표 지표(혼동행렬·클래스별 표)는 값이 없어 지표 목록에 싣지 않는다(옛 요청은 값 0 줄로 온다).
TABLE_METRICS = frozenset({"M21", "M22"})

# 글 속 지표 번호 — 지시문에는 지표 번호를 싣지 않는다(재-4). 고정 문구 18 의 'Accuracy(M1)' 은 이름만 남기고,
# 신뢰구간 방식 한 줄의 'Wilson 점수 구간(M1)'·'…; M4, M5)' 처럼 번호만 있으면 지표 이름으로 바꾼다.
_METRIC_ID = re.compile(r"(?<![A-Za-z0-9])M\d{1,2}(?![A-Za-z0-9])")
_NAMED_ID = re.compile(r"\((M\d{1,2})\)")


def name_metric_ids(text: str, names: dict[str, str], protect: set[str] | frozenset = frozenset()) -> str:
    """지표 번호를 지표 이름으로. '이름(번호)' 꼴은 번호를 지운다. 모르는 번호·클래스 이름(protect)과 같은 글자는 그대로 둔다
    (클래스가 'M1'·'M2' 꼴이면 문제 이름 속 클래스가 지표 이름으로 바뀌던 것 — S5 검토 prompt-5)."""
    for mid, name in names.items():
        text = text.replace(f"{name}({mid})", name)
    return _METRIC_ID.sub(lambda m: m.group(0) if m.group(0) in protect else names.get(m.group(0), m.group(0)), text)


def class_names(fs: FactSheet) -> set[str]:
    """사실 시트에 나오는 클래스·레이블 이름."""
    out = {p.label for p in fs.per_class} | {r.label for r in fs.confusion_counts}
    if fs.confusion:
        out |= set(fs.confusion.labels)
    if fs.distribution:
        out |= set(fs.distribution.class_distribution)
    if fs.imbalance_effect:
        out |= {fs.imbalance_effect.majority_class, fs.imbalance_effect.minority_class}
    if fs.facts is not None:
        out |= {r.label for r in fs.facts.distribution_shift}
        if fs.facts.weakest_class:
            out.add(fs.facts.weakest_class.label)
    return out


def strip_metric_ids(text: str) -> str:
    """이름 뒤 '(Mn)' 만 지운다(이름을 모를 때)."""
    return _NAMED_ID.sub("", text)


def metric_names(fs: FactSheet) -> dict[str, str]:
    return {m.metric_id: m.display_name for m in fs.metrics}


def _yes(flag: bool | None) -> str | None:
    if flag is None:
        return None
    return "예" if flag else "아니오"


def _r(value: Any) -> Any:
    """판정과 무관한 실수는 소수 넷째 자리로(허용 목록이 받는 표기). 정수·글은 그대로."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    rounded = round(value, 4)
    return int(rounded) if rounded == int(rounded) and abs(rounded) < 1e15 else rounded


def _clean(obj: Any) -> Any:
    """None·빈 칸을 뺀다(지시문을 짧게 — '없음'을 뜻하는 칸은 호출부가 글로 적는다)."""
    if isinstance(obj, dict):
        out = {k: _clean(v) for k, v in obj.items() if v is not None}
        return {k: v for k, v in out.items() if v not in ({}, [], "")}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _overview(fs: FactSheet, task_label: str, report_purpose: str) -> dict:
    facts = fs.facts
    return {
        "분류 유형": task_label,
        "성적서 용도": (facts.report_purpose_label if facts and facts.report_purpose_label else None)
        or REPORT_PURPOSE_KO.get(report_purpose, report_purpose),
        "평가 표본 수": fs.n_samples,
        "전처리에서 제외된 행 수": fs.dropped_rows,
        "양성 클래스": (facts.positive_class if facts else None)
        or (fs.confusion.positive_class if fs.confusion else None),
        "정답 클래스 수": facts.n_classes if facts else None,
        "클래스 비율의 기준": (
            ("레이블 등장 횟수" if facts.class_share_basis == "label_occurrences" else "전체 표본") if facts else None
        ),
    }


def _names(refs) -> list[str]:
    return [r.display_name for r in refs]


def _verdict(fs: FactSheet, derived: dict) -> dict:
    facts = fs.facts
    vd = facts.verdict_detail if facts else None
    out: dict[str, Any] = {"종합 판정": VERDICT_KO.get(fs.verdict, fs.verdict)}
    if facts is not None and facts.core_metrics_source is not None:
        out["핵심 지표를 정한 쪽"] = (
            "의뢰자가 지정함" if facts.core_metrics_source == "client" else "분류 유형의 기본값(의뢰자가 지정하지 않음)"
        )
    if vd is not None:
        out["판정 대상 지표 수"] = vd.judged_count
        out["목표를 충족한 지표 수(참고)"] = vd.passed_count
        out["미달한 핵심 지표"] = _names(vd.failed_core)
        out["미달한 비핵심 지표"] = _names(vd.failed_noncore)
        out["산출하지 못한 핵심 지표"] = _names(vd.unavailable_core)
        if vd.inconclusive_cause:
            out["판정 불가 원인"] = CAUSE_KO[vd.inconclusive_cause]
            out["판정 불가 원인 지표"] = _names(vd.inconclusive_metrics)
    else:
        counts = derived.get("counts") or {}
        out["판정 대상 지표 수"] = counts.get("target")
        out["목표를 충족한 지표 수(참고)"] = counts.get("passed")
    out["통과율(참고, %)"] = fs.score
    return out


def _metrics(fs: FactSheet) -> list[dict]:
    rows = []
    for m in fs.metrics:
        if m.metric_id in TABLE_METRICS:
            continue
        judged = m.threshold is not None
        row: dict[str, Any] = {
            "지표": m.display_name,
            # 판정 대상 값은 인쇄 표기(성적서 6.1 과 같음) — 없으면 원래 값(반올림하면 판정과 어긋나 보일 수 있다).
            "값": m.value_text if m.value_text else (m.value if judged else _r(m.value)),
            "목표값": m.threshold if judged else "없음",
            "결과": STATUS_KO.get(m.status, m.status),
            "핵심 지표": "예" if m.is_core else "아니오",
            "좋은 방향": None if m.higher_is_better is None else ("높을수록 좋음" if m.higher_is_better else "낮을수록 좋음"),
            "β": m.beta,
        }
        if m.margin is not None:
            row["목표와의 차이(0 이상이면 충족 쪽)"] = m.margin_text if m.margin_text else m.margin
        if m.near_target:
            row["근소 차이(목표값의 3% 미만)"] = "예"
        if m.ci_low is not None and m.ci_high is not None:
            row["95% 신뢰구간"] = f"{m.ci_low:g}~{m.ci_high:g}"
            contains = m.ci_contains_target
            if contains is not None:
                row["구간이 목표값을 포함"] = _yes(contains)
        rows.append(row)
    return rows


def _confusion(fs: FactSheet, derived: dict, task_type: str) -> dict | None:
    conf = derived.get("confusion")
    out: dict[str, Any] = {}
    if conf:
        cell = conf.get("unit") == "label_cell"
        totals: dict[str, Any] = {
            "단위": "레이블 칸(표본 수 × 레이블 수)" if cell else "표본",
            "전체": conf.get("total"),
            "맞음": conf.get("correct"),
            "틀림": conf.get("misclassified"),
            "맞음 비율(%)": conf.get("correct_pct"),
            "틀림 비율(%)": conf.get("misclassified_pct"),
        }
        for key in ("tp", "fp", "fn", "tn"):
            if key in conf:
                totals[key.upper()] = conf[key]
                totals[f"{key.upper()} 비율(%)"] = conf.get(f"{key}_pct")
        out["합계"] = totals
    if fs.confusion and fs.confusion.matrix:
        out["행렬(행 = 실제, 열 = 예측)"] = {"클래스 순서": fs.confusion.labels, "행렬": fs.confusion.matrix}
    if fs.confusion_counts:
        out[
            "양성 클래스 기준 TP·FP·FN·TN" if task_type == "binary" else "클래스별 TP·FP·FN·TN(그 클래스 대 나머지)"
        ] = [{"클래스": r.label, "TP": r.tp, "FP": r.fp, "FN": r.fn, "TN": r.tn} for r in fs.confusion_counts]
    return out or None


def _per_class(fs: FactSheet) -> dict | None:
    out: dict[str, Any] = {}
    if fs.per_class:
        out["클래스별"] = [
            {"클래스": p.label, "Precision": _r(p.precision), "Recall": _r(p.recall), "F1": _r(p.f1), "표본 수": p.support}
            for p in fs.per_class
        ]
    if fs.class_averages:
        out["평균 줄"] = [
            {"평균": a.label, "Precision": _r(a.precision), "Recall": _r(a.recall), "F1": _r(a.f1), "표본 수": a.support}
            for a in fs.class_averages
        ]
    return out or None


def _distribution(fs: FactSheet, derived: dict, task_type: str) -> dict | None:
    out: dict[str, Any] = {}
    dist = derived.get("distribution")
    if fs.distribution and fs.distribution.class_distribution:
        out["클래스별 건수"] = fs.distribution.class_distribution
        if dist:
            out["백분율(%, 분모는 평가 표본 수)"] = dist.get("percentages")
    if fs.distribution and fs.distribution.imbalance_ratio is not None:
        # 1.5 와 비교되는 값이다 — 엔진 표시 문자열이 있으면 그것을, 없으면 원래 값(반올림하면 1.49996 이 1.5 가 된다).
        ie_shown = fs.imbalance_effect.display if fs.imbalance_effect and fs.imbalance_effect.display else None
        out["불균형 비율"] = ie_shown.imbalance_ratio if ie_shown else fs.distribution.imbalance_ratio
    if task_type == "multilabel" and out:
        out["유의"] = "레이블 등장 횟수이다. 한 표본에 레이블이 여럿일 수 있어 백분율을 모두 더하면 전체를 넘을 수 있다."
    ie = fs.imbalance_effect
    if ie is not None:
        shown = ie.display
        noun = "레이블" if ie.accuracy_minus_macro_f1 is None else "클래스"
        out["불균형 영향(평가 엔진 판단)"] = {
            "판단": IMBALANCE_KO[ie.level],
            "불균형 비율": shown.imbalance_ratio if shown else ie.imbalance_ratio,
            "정확도 − 매크로 F1": (
                (shown.accuracy_minus_macro_f1 if shown and shown.accuracy_minus_macro_f1 else ie.accuracy_minus_macro_f1)
                if ie.accuracy_minus_macro_f1 is not None else None
            ),
            f"다수 − 소수 {noun} F1 차이": shown.majority_minus_minority_f1 if shown else ie.majority_minus_minority_f1,
            f"다수 {noun}": ie.majority_class,
            f"소수 {noun}": ie.minority_class,
        }
    return out or None


def _error_ratios(fs: FactSheet) -> dict | None:
    er = fs.error_ratios
    if er is None:
        return None
    def part(p):
        return None if p is None else {"건수": p.count, "비율": p.ratio}
    return {
        "단위": "틀린 레이블 칸" if er.unit == "label_cell" else "오분류 표본",
        "확률이 있는 오분류 수": er.misclassified,
        "경계 오류(결정 경계와의 거리가 기준 미만)": part(er.boundary)
        or ("계산하지 않음 — 예측값을 직접 제출해 결정 임계값을 모름" if er.boundary_note == "threshold_unknown" else "계산하지 않음"),
        "확신 오류(틀린 쪽 확률이 기준 이상)": part(er.confident),
        "경계 거리 기준": er.boundary_margin,
        "확신 기준": er.confident_level,
    }


def _threshold_gain(fs: FactSheet, derived: dict) -> dict | None:
    g = fs.f1_threshold_gain
    if g is None:
        return None
    return {
        "현재 F1": g.current_f1,
        "PR 곡선 위 최대 F1": g.max_f1,
        "향상폭(이 데이터에서 오를 수 있는 최대폭)": g.gain_text or g.gain,
        "최대 F1 의 결정 임계값": g.best_threshold,
        "평가 엔진 판단": "현재 임계값이 부적합(향상폭 0.1 이상)" if derived.get("threshold_tuning") else "조정 여지가 크지 않음(향상폭 0.1 미만)",
    }


def _derivation(fs: FactSheet) -> dict | None:
    d = fs.derived_prediction
    if d is None:
        return None
    return {"방법": DERIVATION_KO.get(d.method, "확률에서 예측을 파생함"), "결정 임계값": d.threshold}


def _latency(fs: FactSheet) -> dict | str:
    lat = fs.latency
    if not lat.available:
        return "측정하지 않음 — 지연 수치를 쓰지 않는다"
    return {"평균": lat.mean, "p50": lat.p50, "p95": lat.p95, "p99": lat.p99, "단위": lat.unit}


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
            "F1": _r(w.f1),
            "표본 수": w.support,
            "비율(0~1)": w.share_text,
            "매크로 F1": _r(w.macro_f1),
            "매크로 F1 과의 차이": w.gap_text,
            "표본이 작음(30건 미만 또는 5% 미만)": _yes(w.low_support),
            "매크로 F1 보다 0.1 이상 낮음": _yes(w.far_below_macro),
        }
    if facts.low_support_classes:
        out[f"표본이 작은 {noun}"] = [
            {noun: r.label, "표본 수": r.support, "비율(0~1)": r.share_text} for r in facts.low_support_classes
        ]
    if facts.top_confusion_pairs:
        out["가장 많이 헷갈린 쌍(실제 → 예측)"] = [
            {"실제": p.actual, "예측": p.predicted, "건수": p.count, "전체 오분류 중 비율": _r(p.share_of_errors)}
            for p in facts.top_confusion_pairs
        ]
    bias = facts.prediction_bias
    if bias is not None and bias.rows:
        out["예측 편향"] = {
            "무엇을 비교": BIAS_UNIT_KO[bias.unit],
            "줄": [
                {noun: r.label, "예측 수": r.predicted, "실제 수": r.actual, "차이(예측 − 실제)": r.difference}
                for r in bias.rows
            ],
        }
    dr = facts.declared_risk_errors
    if dr is not None:
        out["의뢰자가 더 위험하다고 밝힌 오류"] = {
            "오류": "FN(놓침)" if dr.side == "fn" else "FP(잘못 잡음)",
            "건수": dr.count,
            "전체 오류 수": dr.total_errors,
            "전체 오류 중 비율(0~1)": dr.share_text or _r(dr.share),
        }
    g = facts.precision_recall_gap
    if g is not None:
        out["양성 클래스 Precision 과 Recall"] = {
            "Precision": _r(g.precision),
            "Recall": _r(g.recall),
            "차이(Precision − Recall)": g.difference_text,
            "치우침(|차이| 0.1 이상)": _yes(g.flagged),
        }
    ag = facts.average_gap
    if ag is not None:
        other = "가중 평균" if ag.kind == "macro_vs_weighted" else "마이크로 평균"
        out[f"매크로 평균과 {other}의 F1 차이"] = {
            "매크로 F1": _r(ag.macro_f1),
            f"{other} F1": _r(ag.other_f1),
            "차이": ag.difference_text,
            "기준 클래스" if ag.kind == "macro_vs_weighted" else "F1 이 가장 낮은 레이블": {
                noun: ag.reference_class.label,
                "표본 수": ag.reference_class.support,
                "F1": _r(ag.reference_class.f1),
            },
        }
    da = facts.declared_average
    if da is not None:
        out["의뢰자 선언에 맞춘 평균 방식의 결과"] = {
            "선언": PRIORITY_DECL_KO[da.class_priority],
            "평균 방식": "가중 평균" if da.average == "weighted" else "마이크로 평균",
            "Precision": _r(da.precision),
            "Recall": _r(da.recall),
            "F1": _r(da.f1),
        }
    out["학습↔평가 분포 비교"] = SHIFT_STATUS_KO[facts.distribution_shift_status]
    if facts.distribution_shift:
        out["학습↔평가 분포 차이"] = [
            {
                noun: r.label,
                "학습 비율": _r(r.training_ratio),
                "평가 비율": _r(r.evaluation_ratio),
                "차이": r.difference_text,
                "10%p 이상": _yes(r.flagged),
            }
            for r in facts.distribution_shift
        ]
    if facts.ci_skipped:
        out["신뢰구간을 산출하지 않은 지표"] = [
            {"지표": s.display_name, "사유": name_metric_ids(s.reason, metric_names(fs), class_names(fs)) if s.reason else "제공되지 않음"}
            for s in facts.ci_skipped
        ]
    decl = facts.declarations
    out["의뢰자 선언"] = {
        "위험한 오류 방향": RISK_KO.get(decl.risk_direction) if decl.risk_direction else None,
        "클래스 중요도": PRIORITY_DECL_KO.get(decl.class_priority) if decl.class_priority else None,
        "사용 방식": USAGE_KO.get(decl.usage_mode) if decl.usage_mode else None,
    }
    return out


def _benchmarks(benchmark_refs: list[dict]) -> dict | None:
    if not benchmark_refs:
        return None
    return {
        "기준의 출처·성격": benchmark_refs[0].get("source_note"),
        "지표별": [
            {
                "지표": r["metric"],
                "모델 값": r["model_value"],
                "참조 범위": f"{r['ref_low']:g}~{r['ref_high']:g}",
                "좋은 방향": "높을수록 좋음" if r.get("direction") == "higher" else "낮을수록 좋음",
                "평가(방향 반영)": QUALITY_KO.get(r.get("quality"), r.get("quality")),
                "목표값": r.get("target_value"),
                "목표값의 위치": TARGET_POSITION_KO.get(r.get("target_position")) if r.get("target_position") else None,
            }
            for r in benchmark_refs
        ],
    }


def translate_evidence(obj: Any) -> Any:
    """문제 근거의 키·코드값을 한국어로. 모르는 키는 그대로 둔다(테스트가 계약 예시에서 영어 키가 남지 않음을 본다)."""
    if isinstance(obj, bool):
        return "예" if obj else "아니오"
    if isinstance(obj, dict):
        shown = obj.get("display") if isinstance(obj.get("display"), dict) else {}
        if isinstance(obj.get("gain_text"), str):
            shown = {**shown, "gain": obj["gain_text"]}
        obj = {k: (shown[k] if k in shown and shown[k] is not None else v) for k, v in obj.items()
               if k not in ("display", "gain_text")}
        out = {}
        for key, value in obj.items():
            if key in EVIDENCE_SKIP:
                continue
            if key == "level" and value in IMBALANCE_KO:
                out["판단"] = IMBALANCE_KO[value]
                continue
            out[EVIDENCE_KEYS_KO.get(key, key)] = translate_evidence(value)
        return out
    if isinstance(obj, list):
        return [translate_evidence(v) for v in obj]
    return obj


def _problems(fs: FactSheet) -> list[dict]:
    return [
        {
            "번호": p.id,
            "우선순위": PRIORITY_KO[p.priority],
            "문제": p.title,  # 프론트 문제 목록은 지표 이름으로 쓴다 — 번호를 바꾸지 않는다(클래스 'M1' 보호)
            # '8.2절'처럼 '절'을 붙인다 — 서술이 옮겨 적어도 절 번호로 읽힌다(숫자 검문의 절 번호 규칙).
            "드러난 절": f"{p.source_section}절" if p.source_section else None,
            "권고 절": f"{p.recommend_section}절" if p.recommend_section else None,
            "근거": translate_evidence(p.evidence) or None,
            "덧붙임": p.note,
        }
        for p in fs.problems
    ]


def omitted_counts(fs: FactSheet) -> dict[str, int]:
    """프론트가 표를 자른 수(facts.omitted) — 한국어 이름으로."""
    if fs.facts is None:
        return {}
    # 모르는 키는 영어로 흘리지 않는다(S5 검토 prompt-3).
    return {OMITTED_KO.get(k, "그 밖의 표"): v for k, v in fs.facts.omitted.items() if v}


def build_fact_view(
    fs: FactSheet, derived: dict, benchmark_refs: list[dict], *, task_type: str, task_label: str, report_purpose: str,
) -> dict:
    """AI 가 읽는 사실 시트(한국어 이름). None·빈 칸은 뺀다."""
    view = {
        "평가 개요": _overview(fs, task_label, report_purpose),
        "판정": _verdict(fs, derived),
        "지표": _metrics(fs),
        "혼동행렬": _confusion(fs, derived, task_type),
        "클래스별 성능": _per_class(fs),
        "분포·불균형": _distribution(fs, derived, task_type),
        "오분류의 성격": _error_ratios(fs),
        "임계값 조정 여지": _threshold_gain(fs, derived),
        "예측 파생": _derivation(fs),
        "지연시간": _latency(fs),
        "신뢰구간 방식": name_metric_ids(fs.ci_method_note, metric_names(fs), class_names(fs)) if fs.ci_method_note else None,
        "서술용 파생값": _facts(fs, task_type),
        "벤치마크(내부 참조 기준)": _benchmarks(benchmark_refs),
        "문제 목록": _problems(fs),
        "덜어 낸 표(일부만 실림)": omitted_counts(fs),
    }
    return _clean(view)


def build_client_view(client_inputs: dict | None) -> tuple[dict | None, list[str]]:
    """고객 입력을 한국어 묶음 이름으로. 모르는 묶음은 싣지 않고 그 키를 돌려준다(내부 기록 — D34)."""
    if not client_inputs:
        return None, []
    view: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in client_inputs.items():
        name = CLIENT_GROUPS_KO.get(key)
        if name is None:
            dropped.append(str(key))
            continue
        view[name] = copy.deepcopy(value)
    return (_clean(view) or None), dropped


# ── 지시문 크기 상한에 맞춰 덜어 내기 ───────────────────────────────────────────
# 클래스가 많으면 표가 지시문을 키운다. 결론(최약 클래스·최다 오분류 쌍)은 결론 목록·서술용 파생값에 이미 있어 큰 표부터
# 뺀다(옛 경로 build_prompts_within_limit 과 같은 차례). 숫자 허용 목록은 원본 사실 시트로 만든다 — 뺀 표의 숫자를
# 인용해도 사실이다. 뺀 칸은 보기의 '덜어 낸 표'에 더한다(빈 목록을 '없다'로 읽지 않게 — S4b 검토 SV-01).


def _note(view: dict, name: str, n: int) -> None:
    if n > 0:
        bucket = view.setdefault("덜어 낸 표(일부만 실림)", {})
        bucket[name] = bucket.get(name, 0) + n


def _drop_matrix(view: dict) -> bool:
    conf = view.get("혼동행렬") or {}
    matrix = conf.pop("행렬(행 = 실제, 열 = 예측)", None)
    if matrix is None:
        return False
    _note(view, OMITTED_KO["confusion"], len(matrix.get("클래스 순서") or []))
    return True


def _drop_per_class(view: dict) -> bool:
    table = (view.get("클래스별 성능") or {}).pop("클래스별", None)
    if table is None:
        return False
    _note(view, OMITTED_KO["per_class"], len(table))
    return True


def _drop_counts(view: dict) -> bool:
    conf = view.get("혼동행렬") or {}
    for key in list(conf):
        if "TP·FP·FN·TN" in key:
            _note(view, OMITTED_KO["confusion_counts"], len(conf.pop(key)))
            return True
    return False


def _drop_distribution(view: dict) -> bool:
    dist = view.get("분포·불균형") or {}
    counts = dist.pop("클래스별 건수", None)
    dist.pop("백분율(%, 분모는 평가 표본 수)", None)
    if counts is None:
        return False
    _note(view, OMITTED_KO["class_distribution"], len(counts))
    return True


def _drop_fact_lists(view: dict) -> bool:
    facts = view.get("서술용 파생값") or {}
    n = 0
    for key in list(facts):
        if key.startswith("표본이 작은") or key == "학습↔평가 분포 차이":
            n += len(facts.pop(key))
        elif key == "예측 편향":
            n += len(facts[key].get("줄", []))
            facts.pop(key)
    if n == 0:
        return False
    _note(view, OMITTED_KO["facts_class_lists"], n)
    return True


TRIM_STEPS = (
    ("confusion.matrix", _drop_matrix),
    ("per_class", _drop_per_class),
    ("confusion_counts", _drop_counts),
    ("distribution", _drop_distribution),
    ("facts.class_lists", _drop_fact_lists),
)
