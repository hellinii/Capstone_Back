"""app/narrative/derived.py — 서버 파생 사실(순수)

서버가 계산하는 것은 혼동행렬 합계·백분율(멀티레이블은 레이블 칸 단위)과 클래스 분포 백분율뿐이다(계획 설계
요지 4). 판정 카운트·구간이 목표를 포함하는 지표·오분류 총건수(행렬이 잘렸을 때)는 프론트 서술용 파생값
(fact_sheet.facts, metrics[].ci_contains_target)을 옮긴다 — 같은 숫자를 두 곳에서 세지 않는다. facts 가 없는
옛 요청(운영 main 프론트, 저장된 다시 만들기 요청)에만 지금까지의 서버 셈을 쓴다.
LLM 은 계산하지 않으며 이 값들만 인용한다.

상호작용
- 의존(import): app.core.schemas(FactSheet)
- 사용처: app.narrative.legacy·generator·required_items·fallback(서술 전 파생값 준비)
"""
from app.narrative.schemas import FactSheet

# 임계값 조정 여지가 '부적합'으로 볼 만큼 큰지의 기준(서술 설계 D40) — notable 이 없는 옛 요청에만 쓴다.
# 숫자는 엔진 한 곳(metrics.binary)에서 가져온다.
from app.evaluation.metrics.binary import THRESHOLD_GAIN_NOTABLE  # noqa: E402


def _threshold_tuning(gain) -> bool:
    """임계값 부적합인지 — 평가 엔진이 반올림 전 값으로 정한 notable(S2)을 옮긴다. 판정은 엔진 한 곳이다.

    notable 이 없는 옛 요청(운영 main 프론트가 f1_threshold_gain 을 notable 없이 옮김)만 실린 4자리 gain ≥ 0.10 으로
    본다. 지시문 규칙 14 와 폴백 8.3 은 이 값을 compute_derived 의 threshold_tuning 으로 읽는다(따로 판정하지 않음).
    """
    if gain.notable is not None:
        return gain.notable
    return gain.gain >= THRESHOLD_GAIN_NOTABLE


def _find_pos_idx(labels: list, positive_class) -> int:
    """2x2 혼동행렬에서 양성 클래스의 행/열 index(0|1)를 찾는다.

    labels 는 정렬된 클래스 문자열, positive_class 는 metadata 의 양성값.
    직접 → 정규화(strip/casefold) → float 비교 순으로 매칭하고("1.0" vs "1" 등),
    못 찾으면 index 1 로 폴백한다(평가기 'sorted-last=positive' 규칙 = 기존 동작 하위호환).
    """
    if not positive_class or not labels or len(labels) != 2:
        return 1
    norm = [str(l).strip() for l in labels]
    p = str(positive_class).strip()
    if p in norm:
        return norm.index(p)
    low = [x.casefold() for x in norm]
    if p.casefold() in low:
        return low.index(p.casefold())
    try:
        pf = float(p)
        floats = [float(x) for x in norm]
        if pf in floats:
            return floats.index(pf)
    except ValueError:
        pass
    return 1


def _confusion_from_counts(rows, task_type: str | None) -> dict | None:
    """엔진 클래스별 건수로 만든 혼동행렬 합계(행렬이 없는 옛 요청). 작업 유형을 모르면 줄이 하나일 때만(이진) 센다 —
    다중과 멀티레이블은 분모가 달라(표본 / 레이블 칸) 짐작하지 않는다."""
    kind = task_type or ("binary" if len(rows) == 1 else None)
    if kind == "binary" and len(rows) == 1:
        r = rows[0]
        total, misclassified = r.tp + r.fp + r.fn + r.tn, r.fp + r.fn
        conf = {"total": total, "correct": total - misclassified, "misclassified": misclassified,
                "tn": r.tn, "fp": r.fp, "fn": r.fn, "tp": r.tp}
    elif kind == "multiclass":
        total = sum(r.tp + r.fn for r in rows)
        misclassified = sum(r.fn for r in rows)
        conf = {"total": total, "correct": total - misclassified, "misclassified": misclassified}
    elif kind == "multilabel":
        total = sum(r.tp + r.fp + r.fn + r.tn for r in rows)
        misclassified = sum(r.fp + r.fn for r in rows)
        conf = {"total": total, "correct": total - misclassified, "misclassified": misclassified, "unit": "label_cell"}
    else:
        return None
    if total > 0:
        conf["correct_pct"] = round(conf["correct"] / total * 100, 1)
        conf["misclassified_pct"] = round(misclassified / total * 100, 1)
    return conf


def compute_derived(fs: FactSheet, *, task_type: str | None = None) -> dict:
    """LLM·폴백이 공통으로 인용할 파생 사실(서버 계산). 환각 방지를 위해 덧셈은 여기서만.

    task_type 이 "multilabel" 이면 혼동행렬 합계의 단위가 표본이 아니라 **레이블 칸**이다(프론트가 레이블별
    2×2 를 더해 보낸다 — 표본 200개·레이블 4개면 800칸). 그 단위를 confusion.unit = "label_cell" 로 싣고,
    분포에는 unit = "label" 을 싣는다(한 표본에 레이블이 여럿이라 비율 합이 100% 를 넘는다).
    None(옛 호출)이면 종전과 같다.
    """
    derived: dict = {}
    multilabel = task_type == "multilabel"

    if fs.confusion and fs.confusion.matrix:
        m = fs.confusion.matrix
        total = sum(sum(row) for row in m)
        correct = sum(m[i][i] for i in range(len(m)) if i < len(m[i]))
        misclassified = total - correct
        conf = {
            "total": total,
            "correct": correct,
            "misclassified": misclassified,
        }
        if multilabel:
            conf["unit"] = "label_cell"
        # 흔히 인용되는 파생 백분율도 서버가 미리 계산(정확값)해 화이트리스트에 포함시킨다.
        # → LLM 이 오분류율 등을 인용해도 grounding 통과(틀린 비율은 여전히 차단됨).
        if total > 0:
            conf["correct_pct"] = round(correct / total * 100, 1)
            conf["misclassified_pct"] = round(misclassified / total * 100, 1)
        # 2x2(binary)면 FN/FP 및 그 비율도 제공 (행=실제, 열=예측).
        # positive 클래스의 실제 index 를 근거로 매핑한다(index 1 하드코딩 금지).
        if len(m) == 2 and len(m[0]) == 2:
            pos_idx = _find_pos_idx(fs.confusion.labels, fs.confusion.positive_class)
            neg_idx = 1 - pos_idx
            tp = m[pos_idx][pos_idx]
            tn = m[neg_idx][neg_idx]
            fp = m[neg_idx][pos_idx]  # 실제=음성, 예측=양성
            fn = m[pos_idx][neg_idx]  # 실제=양성, 예측=음성
            conf.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
            if total > 0:
                conf.update({
                    "tn_pct": round(tn / total * 100, 1),
                    "fp_pct": round(fp / total * 100, 1),
                    "fn_pct": round(fn / total * 100, 1),
                    "tp_pct": round(tp / total * 100, 1),
                })
        derived["confusion"] = conf
    elif fs.facts is not None and fs.facts.error_totals is not None:
        # 클래스가 많아 혼동행렬이 잘렸어도(프론트 미리 자르기) 오분류 총건수는 자르기 전 엔진 건수로 남는다.
        totals = fs.facts.error_totals
        conf = {
            "total": totals.total,
            "correct": totals.total - totals.misclassified,
            "misclassified": totals.misclassified,
        }
        if totals.unit == "label_cell":
            conf["unit"] = "label_cell"
        if totals.total > 0:
            conf["correct_pct"] = round(conf["correct"] / totals.total * 100, 1)
            conf["misclassified_pct"] = round(totals.misclassified / totals.total * 100, 1)
        derived["confusion"] = conf
    elif fs.confusion_counts:
        # 옛 요청(운영 main 프론트)에서 혼동행렬(M21)을 고르지 않았으면 행렬이 없다 — 엔진 클래스별 건수(confusion_counts,
        # M21 선택과 무관하게 실림)로 오분류 총건수를 센다(계약 error_totals 와 같은 규칙: 이진 fp+fn / 전체, 다중 Σfn / Σ(tp+fn),
        # 멀티 Σ(fp+fn) / Σ(tp+fp+fn+tn) 레이블 칸). 옛 요청만의 서버 셈이다(새 요청은 facts.error_totals 를 옮긴다).
        conf = _confusion_from_counts(fs.confusion_counts, task_type)
        if conf is not None:
            derived["confusion"] = conf

    if fs.distribution and fs.distribution.class_distribution:
        dist = fs.distribution.class_distribution
        # 분모는 '표본 수'이지 '등장 횟수 합'이 아니다(ISSUES.md B-02). 멀티레이블에서
        # 둘은 갈린다 — 200행이 408 로 서술되던 원인이다. binary/multiclass 는
        # value_counts() 가 프레임 행을 그대로 세므로 두 값이 구조적으로 같아
        # task_type 을 알 필요가 없다. n_samples 가 없는 구 클라이언트는 종전대로 합계.
        # 멀티레이블에서는 백분율 합이 100% 를 넘을 수 있다(buildDatasetDiagnosis 와 같은 규약).
        total = fs.n_samples if fs.n_samples > 0 else sum(dist.values())
        derived["distribution"] = {
            **({"unit": "label"} if multilabel else {}),
            "total": total,
            "percentages": {
                k: round((v / total) * 100, 1) if total > 0 else 0.0
                for k, v in dist.items()
            },
        }

    # 판정 카운트(통과율 근거, 참고 수치) — 폴백/LLM 이 "N개 중 M개 통과"를 인용할 때 사용.
    # 새 요청은 프론트 판정(computeVerdict)의 판정 대상 수·통과 수를 옮긴다. 옛 요청만 서버가 센다 —
    # 목표값 0·음수(MCC 등)도 목표다("없음"은 None 뿐, 프론트 PR #55 의 null 규칙과 같다).
    verdict_detail = fs.facts.verdict_detail if fs.facts is not None else None
    if verdict_detail is not None:
        target, passed = verdict_detail.judged_count, verdict_detail.passed_count
    else:
        target = sum(1 for m in fs.metrics if m.threshold is not None)
        passed = sum(1 for m in fs.metrics if m.status == "pass" and m.threshold is not None)
    derived["counts"] = {
        "target": target,
        "passed": passed,
        "n_metrics": len(fs.metrics),
    }

    # 구간이 목표값을 포함하는 지표(서술 설계 D29). 목표값이 구간 안(양 끝 포함)이면 표본이 달라졌을 때 그 지표의
    # 판정이 바뀔 수 있다. 새 요청은 프론트가 반올림 전 값으로 정한 ci_contains_target 을 옮기고, 그 칸이 없는
    # 옛 요청만 실린 값으로 본다. 신뢰구간이 하나도 실리지 않은 사실 시트에는 키를 두지 않는다.
    if any(m.ci_low is not None and m.ci_high is not None for m in fs.metrics):
        derived["ci_contains_target"] = [
            m.display_name
            for m in fs.metrics
            if (
                m.ci_contains_target
                if m.ci_contains_target is not None
                else m.threshold is not None
                and m.ci_low is not None
                and m.ci_high is not None
                and m.ci_low <= m.threshold <= m.ci_high
            )
        ]

    # 임계값 조정 여지(D40) — 키가 실린 사실 시트에만 둔다(옛 요청은 derived 가 그대로).
    if fs.f1_threshold_gain is not None:
        derived["threshold_tuning"] = _threshold_tuning(fs.f1_threshold_gain)

    return derived
