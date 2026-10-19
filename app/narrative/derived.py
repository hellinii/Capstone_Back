"""app/narrative/derived.py — 서버 파생 사실 계산(순수)

혼동행렬 파생(정오분류·TP/TN/FP/FN), 클래스 분포 백분율, 임계값 지표 판정 카운트,
신뢰구간이 목표값을 포함하는 지표 목록, 임계값 조정 여지가 큰지를 fact_sheet 로부터 미리 계산한다.
LLM 은 계산하지 않으며 이 값들만 인용한다.

상호작용
- 의존(import): app.core.schemas(FactSheet)
- 사용처: app.narrative.service(generate_narrative 가 서술 전 파생값 준비)
"""
from app.narrative.schemas import FactSheet

# 임계값 조정 여지가 '부적합'으로 볼 만큼 큰지의 기준(서술 설계 D40) — notable 이 없는 옛 요청에만 쓴다.
# 숫자는 엔진 한 곳(metrics.binary)에서 가져온다.
from app.evaluation.metrics.binary import THRESHOLD_GAIN_NOTABLE  # noqa: E402


def threshold_tuning_needed(fs: FactSheet) -> bool:
    """임계값 부적합인지 — 평가 엔진이 반올림 전 값으로 정한 notable 을 읽는다(S2, 프론트 문제 목록과 같은 값).

    notable 이 없는 옛 요청만 실린 4자리 gain ≥ 0.10 으로 판정한다(그때는 0.09996 이 0.1 로 반올림돼 걸린다).
    S4b 에서 이 함수를 지우고 엔진 notable 만 읽는다(계획 설계 요지 4).
    """
    gain = fs.f1_threshold_gain
    if gain is None:
        return False
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

    # 판정 카운트(통과율 근거) — 폴백/LLM 이 "N개 중 M개 통과"를 인용할 때 사용
    # 목표값 0·음수(MCC 등)도 목표다 — "없음"은 None 뿐이다(프론트 PR #55 의 null 규칙과 같다).
    target = sum(1 for m in fs.metrics if m.threshold is not None)
    passed = sum(1 for m in fs.metrics if m.status == "pass" and m.threshold is not None)
    derived["counts"] = {
        "target": target,
        "passed": passed,
        "n_metrics": len(fs.metrics),
    }

    # 구간이 목표값을 포함하는 지표(서술 설계 D29) — 판정은 여기 한 곳에서만 한다.
    # 목표값이 구간 안(양 끝 포함)이면 표본이 달라졌을 때 그 지표의 판정이 바뀔 수 있다.
    # 신뢰구간이 하나도 실리지 않은 사실 시트(옛 요청)에는 키를 두지 않는다 — 프롬프트·폴백이 그대로다.
    if any(m.ci_low is not None and m.ci_high is not None for m in fs.metrics):
        derived["ci_contains_target"] = [
            m.display_name
            for m in fs.metrics
            if m.threshold is not None
            and m.ci_low is not None
            and m.ci_high is not None
            and m.ci_low <= m.threshold <= m.ci_high
        ]

    # 임계값 조정 여지(D40) — 키가 실린 사실 시트에만 둔다(옛 요청은 derived 가 그대로).
    if fs.f1_threshold_gain is not None:
        derived["threshold_tuning"] = threshold_tuning_needed(fs)

    return derived
