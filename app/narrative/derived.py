"""app/narrative/derived.py — 서버 파생 사실 계산(순수)

혼동행렬 파생(정오분류·TP/TN/FP/FN), 클래스 분포 백분율, 임계값 지표 판정 카운트를
fact_sheet 로부터 미리 계산한다. LLM 은 계산하지 않으며 이 값들만 인용한다.

상호작용
- 의존(import): app.core.schemas(FactSheet)
- 사용처: app.narrative.service(generate_narrative 가 서술 전 파생값 준비)
"""
from app.narrative.schemas import FactSheet


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


def compute_derived(fs: FactSheet) -> dict:
    """LLM·폴백이 공통으로 인용할 파생 사실(서버 계산). 환각 방지를 위해 덧셈은 여기서만."""
    derived: dict = {}

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
            "total": total,
            "percentages": {
                k: round((v / total) * 100, 1) if total > 0 else 0.0
                for k, v in dist.items()
            },
        }

    # 판정 카운트(통과율 근거) — 폴백/LLM 이 "N개 중 M개 통과"를 인용할 때 사용
    target = sum(1 for m in fs.metrics if m.threshold)
    passed = sum(1 for m in fs.metrics if m.status == "pass" and m.threshold)
    derived["counts"] = {
        "target": target,
        "passed": passed,
        "n_metrics": len(fs.metrics),
    }

    return derived
