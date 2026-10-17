"""app/evaluation/metrics/intervals.py — 지표 95% 신뢰구간(서술 설계 D29)

6.1·6.2 의 "95% 신뢰구간" 열과 1.4 고정 문구 3 을 채운다. 지표가 아니라 success_metrics 의
confidence_intervals 키로 내려보낸다. 엔진은 **고른 지표 전부(M21·M22 제외)** 를 계산하고,
판정 대상에만 붙이는 일은 프론트가 한다. 계산에 실패한 지표에는 구간을 두지 않는다.

지표 → 방식(값마다 한 곳):
- 비율 지표 → Wilson 점수 구간: M1, 이진 M2·M3·M7·M8, M12(다중 마이크로 = 정확도), M16.
- 혼동행렬로 정해지는 나머지 → 혼동행렬 칸 다항 재표집(`bootstrap_multinomial`). 행 수와 무관하게
  칸 수만큼만 일한다. M23 은 정답 클래스 수를 다항 재표집한다.
- 점수 지표(M9·M10·M19)와 멀티레이블 지표 → 작업량 상한 안이면 행 재표집(`bootstrap`).
  넘으면 해석적 방법: AUROC DeLong, AUPRC 로짓, 표본 평균 꼴(로그 손실·해밍·자카드)은 정규근사.
  그래도 안 되면 `skipped` + 사유.

재표집은 2,000회, 지표마다 `default_rng(4213)` 을 새로 만들어 결정적이다. 구간은 백분위 2.5/97.5.
재표집 통계는 원 지표 함수와 **같은 정의**를 가중치(재표집 횟수)로 다시 쓴 것이다 — sklearn 을
2,000번 부르면 수 초가 아니라 수 분이 걸린다. 정의가 어긋나지 않았는지는 점추정 일치 검사가
지킨다: 같은 식으로 원 데이터에서 다시 낸 값이 엔진 지표 값과 1e-9 넘게 다르면 그 지표는
"점추정 불일치"로 생략한다(틀린 구간을 싣느니 싣지 않는다).

상호작용
- 의존(import): numpy, scipy, .binary, .common, .multilabel
- 사용처: app.evaluation.engine.evaluate (success_metrics 의 confidence_intervals 키)
"""

import logging
import math
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from app.evaluation.errors import METRIC_ERROR_KEY

from . import binary, common, multilabel

logger = logging.getLogger(__name__)

LEVEL = 0.95
Z = NormalDist().inv_cdf(0.5 + LEVEL / 2)  # 1.959963…
N_RESAMPLES = 2000
SEED = 4213
POINT_TOLERANCE = 1e-9

# 재표집 표본 중 지표가 정의되지 않는 것(예: AUROC 인데 양성이 하나도 안 뽑힘)이 이 비율을
# 넘으면 구간을 내지 않는다. 그 아래면 정의되는 표본만으로 백분위를 잡는다.
MAX_UNDEFINED_SHARE = 0.05

# ── 작업량 상한 (실측 근거) ──
# 행 재표집 한 지표의 일 = 재표집 횟수 × 행 수 × 폭(행마다 읽는 칸 수: 점수 지표 1, 멀티레이블은
# 레이블 수). 2026-10-17 개발 노트북 실측: 이진 AUROC·AUPRC 가 일 1.2e8(행 6만)에서 각 약 1.5초,
# 로그 손실 1.1초 — 셋을 모두 고르면 4초를 넘었다. 상한을 그 절반(행 3만)으로 잡아 점수 지표
# 셋을 다 골라도 약 2초다. 멀티레이블은 행렬곱이라 같은 일에서 지표당 0.1초 남짓이다.
# 넘으면 해석적 방법(DeLong·로짓·정규근사)으로 바꾼다 — 행이 많을수록 그 근사는 더 정확하다.
ROW_BOOTSTRAP_WORK_CAP = 60_000_000
# 칸 다항 재표집의 일 = 재표집 횟수 × 0 이 아닌 칸 수. 같은 실측에서 칸 5천 개가 약 1초, 1만 개가
# 3.5초였다(다항 추출이 거의 전부). 재표집 결과는 지표끼리 공유하므로(`_Data.confusion_draws`)
# 다중 클래스 지표를 여럿 골라도 이 비용은 한 번이다. 넘으면(클래스 70여 개가 넓게 섞인 혼동행렬)
# 해석적 대안이 없어 생략한다.
CELL_BOOTSTRAP_WORK_CAP = 10_000_000
# 행 재표집 가중치 행렬을 한 번에 만드는 크기(원소 수). 메모리를 수십 MB 안에 묶는다.
_CHUNK_ELEMENTS = 2_000_000

# M21(혼동행렬)·M22(클래스별 표)는 값 하나가 아니라 구간을 두지 않는다.
EXCLUDED = frozenset({"M21", "M22"})


# ── 결과 모양 ────────────────────────────────────────────────────────────

def _r4(x: float) -> float:
    return round(float(x), 4)


def _interval(low: float, high: float, method: str, **extra) -> Dict[str, Any]:
    return {"low": _r4(low), "high": _r4(high), "method": method, "level": LEVEL, **extra}


def _skipped(reason: str) -> Dict[str, Any]:
    return {"low": None, "high": None, "method": "skipped", "level": LEVEL, "reason": reason}


class _Skip(Exception):
    """구간을 낼 수 없는 사정 — 메시지가 그대로 reason 이 된다."""


def wilson_interval(k: int, n: int) -> Tuple[float, float]:
    """Wilson 점수 구간(95%). n == 0 이면 비율 자체가 없어 _Skip."""
    if n <= 0:
        raise _Skip("분모(해당 표본 수)가 0이라 비율의 구간을 정의할 수 없음")
    p = k / n
    z2 = Z * Z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = Z / denom * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _percentile(samples: np.ndarray, method: str, n_total: int) -> Dict[str, Any]:
    """재표집 통계 → 백분위 구간. 정의되지 않는 표본(nan)이 많으면 _Skip."""
    valid = samples[~np.isnan(samples)]
    if len(valid) < n_total * (1 - MAX_UNDEFINED_SHARE):
        raise _Skip("재표집 표본의 5% 넘게에서 지표가 정의되지 않음(한쪽 클래스가 빠짐)")
    with np.errstate(invalid="ignore"):  # 무한대(M23 에서 클래스가 빠진 표본)끼리의 보간은 nan — 아래에서 거른다
        low, high = np.percentile(valid, [2.5, 97.5])
    if not (np.isfinite(low) and np.isfinite(high)):
        raise _Skip("재표집에서 정답 클래스가 사라지는 경우가 있어 구간 끝을 정할 수 없음")
    return _interval(low, high, method, n_resamples=int(len(valid)), seed=SEED)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """0/0 → 0 (sklearn zero_division=0 과 같다)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape)
    np.divide(num, den, out=out, where=den != 0)
    return out


# ── 혼동행렬 통계 (TP·정답 수·예측 수 벡터로 정해지는 지표) ──────────────────
# 모든 함수는 (재표집 수, 클래스 수) 모양의 tp·t·p 와 표본 수 n 을 받아 (재표집 수,) 를 돌려준다.
# 점추정은 원 혼동행렬을 재표집 1개로 넣어 같은 함수로 낸다.

def _fbeta_per_class(tp, t, p, beta: float):
    b2 = beta * beta
    return _safe_div((1 + b2) * tp, b2 * t + p)


def _masked_mean(values, mask):
    return _safe_div((values * mask).sum(1), mask.sum(1))


def _macro_stat(kind: str, beta: float = 1.0) -> Callable:
    """다중 M2~M5 — 클래스 집합을 정답으로 고정한 macro 평균(common._label_kwargs).

    재표집에서는 그 표본의 정답 클래스로 집합을 다시 정한다(원 지표를 그 표본에 적용한 것과 같다).
    """
    def stat(tp, t, p, n):
        per = {
            "precision": lambda: _safe_div(tp, p),
            "recall": lambda: _safe_div(tp, t),
            "fbeta": lambda: _fbeta_per_class(tp, t, p, beta),
        }[kind]()
        return _masked_mean(per, t > 0)
    return stat


def _m11_stat(tp, t, p, n):
    # M11 은 labels 인자 없이 계산한다 — 정답 ∪ 예측 클래스 전부의 macro F1.
    return _masked_mean(_fbeta_per_class(tp, t, p, 1.0), (t > 0) | (p > 0))


def _m13_stat(tp, t, p, n):
    return _safe_div((_fbeta_per_class(tp, t, p, 1.0) * t).sum(1), t.sum(1))


def _kl_stat(tp, t, p, n):
    # common.calculate_kl_divergence 와 같다: 나타나지 않은 쪽은 1e-9 로 채운다.
    eps = 1e-9
    present = (t > 0) | (p > 0)
    P = np.where(t > 0, t / n, eps)
    Q = np.where(p > 0, p / n, eps)
    return np.where(present, P * np.log(P / Q), 0.0).sum(1)


def _tvd_stat(tp, t, p, n):
    return 0.5 * np.abs(t - p).sum(1) / n


def _mcc_stat(tp, t, p, n):
    # sklearn.matthews_corrcoef 의 다중 클래스 식(이진도 같은 식이다). 분모가 0 이면 0.
    c = tp.sum(1)
    cov_ytyp = c * n - (t * p).sum(1)
    cov_ypyp = n * n - (p * p).sum(1)
    cov_ytyt = n * n - (t * t).sum(1)
    den = cov_ypyp * cov_ytyt
    out = np.zeros(len(c))
    np.divide(cov_ytyp, np.sqrt(np.maximum(den, 0.0)), out=out, where=den > 0)
    return out


def _binary_fbeta_stat(k: Optional[int], beta: float) -> Callable:
    def stat(tp, t, p, n):
        if k is None:
            return np.zeros(len(tp))
        return _fbeta_per_class(tp[:, k], t[:, k], p[:, k], beta)
    return stat


# ── 행 재표집 통계 (가중치 행렬 W: (재표집 수, 행 수) → (재표집 수,)) ───────────

def _score_groups(scores: np.ndarray, y: np.ndarray):
    """점수 오름차순 고유값 묶음 → (행→묶음 희소 행렬: 양성 가중, 음성 가중)."""
    uniq, g = np.unique(scores, return_inverse=True)
    rows = np.arange(len(scores))
    shape = (len(scores), len(uniq))
    pos = sparse.csr_matrix((y.astype(float), (rows, g)), shape=shape)
    neg = sparse.csr_matrix((1.0 - y, (rows, g)), shape=shape)
    return pos, neg


def _auroc_stat(scores: np.ndarray, y: np.ndarray) -> Callable:
    pos_g, neg_g = _score_groups(scores, y)

    def stat(W):
        wp = np.asarray(W @ pos_g)   # (B, U) 묶음별 양성 가중
        wn = np.asarray(W @ neg_g)
        below = np.cumsum(wn, axis=1) - wn   # 그 점수보다 낮은 음성
        num = (wp * (below + 0.5 * wn)).sum(1)
        den = wp.sum(1) * wn.sum(1)
        out = np.full(len(num), np.nan)
        np.divide(num, den, out=out, where=den > 0)
        return out
    return stat


def _auprc_stat(scores: np.ndarray, y: np.ndarray) -> Callable:
    # average_precision_score = Σ (재현율 증가분) × 그 임계값의 정밀도. 임계값은 고유 점수 내림차순.
    pos_g, neg_g = _score_groups(scores, y)

    def stat(W):
        wp = np.asarray(W @ pos_g)[:, ::-1]
        wn = np.asarray(W @ neg_g)[:, ::-1]
        tp = np.cumsum(wp, axis=1)
        fp = np.cumsum(wn, axis=1)
        precision = _safe_div(tp, tp + fp)
        total_pos = wp.sum(1)
        out = np.full(len(total_pos), np.nan)
        np.divide((wp * precision).sum(1), total_pos, out=out, where=total_pos > 0)
        return out
    return stat


def _log_loss_rows(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    # sklearn._log_loss 와 같다: [1−p, p] 를 각각 (eps, 1−eps) 로 자른 뒤 정답 쪽 −log.
    eps = np.finfo(np.float64).eps
    p1 = np.clip(scores, eps, 1 - eps)
    p0 = np.clip(1 - scores, eps, 1 - eps)
    return -np.where(y == 1, np.log(p1), np.log(p0))


def _mean_stat(values: np.ndarray) -> Callable:
    def stat(W):
        return (W @ values) / W.sum(1)
    return stat


def _ml_macro_stat(Yt: np.ndarray, Yp: np.ndarray, kind: str, beta: float = 1.0) -> Callable:
    """멀티레이블 M2~M5 — 정답으로 fit 한 레이블 열의 macro. 재표집 표본의 정답 레이블로 집합을 다시 정한다."""
    both = (Yt & Yp).astype(float)
    Yt_f, Yp_f = Yt.astype(float), Yp.astype(float)

    def stat(W):
        tp, t, p = W @ both, W @ Yt_f, W @ Yp_f
        per = {
            "precision": lambda: _safe_div(tp, p),
            "recall": lambda: _safe_div(tp, t),
            "fbeta": lambda: _fbeta_per_class(tp, t, p, beta),
        }[kind]()
        return _masked_mean(per, t > 0)
    return stat


def _hamming_stat(Yt: np.ndarray, Yp: np.ndarray) -> Callable:
    # 레이블 수는 그 표본에 정답·예측으로 나타난 레이블 수(정답 ∪ 예측으로 fit 하는 원 지표와 같다).
    mism = (Yt != Yp).sum(1).astype(float)
    seen = (Yt | Yp).astype(float)

    def stat(W):
        n_labels = (W @ seen > 0).sum(1)
        return _safe_div(W @ mism, W.sum(1) * n_labels)
    return stat


def _jaccard_rows(Yt: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    inter = (Yt & Yp).sum(1)
    union = (Yt | Yp).sum(1)
    # zero_division=1: 정답·예측이 모두 빈 행은 일치로 센다(multilabel.calculate_jaccard_index).
    return np.where(union == 0, 1.0, _safe_div(inter, union))


def _ml_cosine_stat(Yt: np.ndarray, Yp: np.ndarray) -> Callable:
    Yt_f, Yp_f = Yt.astype(float), Yp.astype(float)

    def stat(W):
        pf, qf = W @ Yt_f, W @ Yp_f
        norm = np.linalg.norm(pf, axis=1) * np.linalg.norm(qf, axis=1)
        out = np.zeros(len(pf))
        np.divide((pf * qf).sum(1), norm, out=out, where=norm > 0)
        return np.where(norm > 0, 1.0 - out, 0.0)
    return stat


def _row_bootstrap(n: int, width: int, stat: Callable) -> Tuple[float, Callable[[], Dict[str, Any]]]:
    """행 재표집. (점추정, 구간을 내는 함수) — 상한을 넘으면 구간 함수 대신 None."""
    point = float(stat(np.ones((1, n)))[0])
    if N_RESAMPLES * n * max(width, 1) > ROW_BOOTSTRAP_WORK_CAP:
        return point, None

    def run() -> Dict[str, Any]:
        rng = np.random.default_rng(SEED)
        chunk = max(1, _CHUNK_ELEMENTS // n)
        out = []
        done = 0
        while done < N_RESAMPLES:
            size = min(chunk, N_RESAMPLES - done)
            idx = rng.integers(0, n, size=(size, n))
            idx += (np.arange(size) * n)[:, None]
            W = np.bincount(idx.ravel(), minlength=size * n).reshape(size, n).astype(float)
            out.append(stat(W))
            done += size
        return _percentile(np.concatenate(out), "bootstrap", N_RESAMPLES)
    return point, run


def _normal_interval(values: np.ndarray, lower: float = 0.0, upper: Optional[float] = None) -> Dict[str, Any]:
    n = len(values)
    if n < 2:
        raise _Skip("표본이 2개 미만이라 정규근사 구간을 계산할 수 없음")
    mean = float(values.mean())
    half = Z * float(values.std(ddof=1)) / math.sqrt(n)
    high = mean + half if upper is None else min(upper, mean + half)
    return _interval(max(lower, mean - half), high, "normal")


def _delong(scores: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """AUROC DeLong 구간(중간 순위로 구조 성분 V10·V01 을 낸다)."""
    pos, neg = scores[y == 1], scores[y == 0]
    m, k = len(pos), len(neg)
    if m < 2 or k < 2:
        raise _Skip("양성 또는 음성 표본이 2개 미만이라 DeLong 분산을 계산할 수 없음")
    all_rank = rankdata(np.concatenate([pos, neg]))
    v10 = (all_rank[:m] - rankdata(pos)) / k
    v01 = 1.0 - (all_rank[m:] - rankdata(neg)) / m
    auc = float(v10.mean())
    se = math.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / k)
    return auc, _interval(max(0.0, auc - Z * se), min(1.0, auc + Z * se), "delong")


def _logit_ap(ap: float, n_pos: int) -> Dict[str, Any]:
    """AUPRC 로짓 구간: logit(AP) ± z / √(양성 수 × AP × (1 − AP))."""
    if not (0.0 < ap < 1.0) or n_pos <= 0:
        raise _Skip("AUPRC 가 0 또는 1이라 로짓 구간을 정의할 수 없음")
    center = math.log(ap / (1 - ap))
    half = Z / math.sqrt(n_pos * ap * (1 - ap))
    expit = lambda x: 1.0 / (1.0 + math.exp(-x))
    return _interval(expit(center - half), expit(center + half), "logit")


# ── 데이터 준비 (지표가 여럿이어도 한 번만) ────────────────────────────────

class _Data:
    def __init__(self, df: pd.DataFrame, mapping_dict: dict):
        self.df = df
        self.m = mapping_dict
        self._cache: Dict[str, Any] = {}

    def _once(self, key: str, make: Callable):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def confusion(self):
        """원래 라벨 값 그대로의 혼동행렬 — (라벨들, 0 아닌 칸의 행·열·건수, n)."""
        def make():
            y_true, y_pred = common._get_true_pred(self.df, self.m)
            n = len(y_true)
            codes, uniques = pd.factorize(
                np.concatenate([np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)])
            )
            if (codes < 0).any():
                raise _Skip("정답·예측에 빈 값이 있어 혼동행렬 재표집을 할 수 없음")
            L = len(uniques)
            cell, counts = np.unique(codes[:n] * L + codes[n:], return_counts=True)
            return list(uniques), cell // L, cell % L, counts, n
        return self._once("confusion", make)

    def label_index(self, value) -> Optional[int]:
        labels = self.confusion()[0]
        for i, u in enumerate(labels):
            if u == value:
                return i
        return None

    def ml_true_fit(self):
        # M2~M5(그리고 M1) 는 정답으로만 fit 한 이진 행렬을 쓴다(common._get_true_pred).
        return self._once("ml_true", lambda: tuple(np.asarray(a, dtype=bool) for a in common._get_true_pred(self.df, self.m)))

    def ml_union_fit(self):
        # M15~M18 은 정답 ∪ 예측으로 fit 한다(multilabel._get_binarized_true_pred).
        return self._once("ml_union", lambda: tuple(np.asarray(a, dtype=bool) for a in multilabel._get_binarized_true_pred(self.df, self.m)))

    def binary_scores(self):
        def make():
            y_true, y_score = binary._get_true_score(self.df, self.m)
            y = np.asarray(binary._binarize_true_labels(y_true, self.m.get("_positive_class")), dtype=float)
            return np.asarray(y_score, dtype=float), y
        return self._once("scores", make)


def _confusion_vectors(data: _Data, draws: np.ndarray):
    """칸 건수 (B, 칸 수) → 클래스별 tp·정답 수·예측 수 (B, L)."""
    labels, rows, cols, _counts, _n = data.confusion()
    L = len(labels)
    nnz = len(rows)
    idx = np.arange(nnz)
    to_true = sparse.csr_matrix((np.ones(nnz), (idx, rows)), shape=(nnz, L))
    to_pred = sparse.csr_matrix((np.ones(nnz), (idx, cols)), shape=(nnz, L))
    diag = rows == cols
    to_tp = sparse.csr_matrix((np.ones(diag.sum()), (idx[diag], rows[diag])), shape=(nnz, L))
    draws = draws.astype(float)
    return np.asarray(draws @ to_tp), np.asarray(draws @ to_true), np.asarray(draws @ to_pred)


def _confusion_counts(data: _Data):
    """원 혼동행렬의 클래스별 (tp, 정답 수, 예측 수, n) — 각각 (L,) 벡터."""
    _labels, _rows, _cols, counts, n = data.confusion()
    tp, t, p = _confusion_vectors(data, counts[None, :])
    return tp[0], t[0], p[0], n


def _confusion_draws(data: _Data):
    """칸 다항 재표집 2,000회의 클래스별 (tp, 정답 수, 예측 수).

    지표마다 `default_rng(4213)` 을 새로 만들면 같은 원 혼동행렬에서 같은 표본이 나온다 — 그래서
    한 번만 뽑아 지표끼리 나눠 쓴다(결과는 지표마다 새로 뽑은 것과 같고, 비용은 한 번이다).
    """
    def make():
        _labels, _rows, _cols, counts, n = data.confusion()
        if N_RESAMPLES * len(counts) > CELL_BOOTSTRAP_WORK_CAP:
            raise _Skip("혼동행렬 칸이 너무 많아 재표집 작업량 상한을 넘음")
        rng = np.random.default_rng(SEED)
        return _confusion_vectors(data, rng.multinomial(n, counts / n, size=N_RESAMPLES))
    return data._once("confusion_draws", make)


def _multinomial_bootstrap(data: _Data, stat: Callable) -> Tuple[float, Callable[[], Dict[str, Any]]]:
    _labels, _rows, _cols, counts, n = data.confusion()
    point = float(stat(*_confusion_vectors(data, counts[None, :]), n)[0])

    def run() -> Dict[str, Any]:
        return _percentile(stat(*_confusion_draws(data), n), "bootstrap_multinomial", N_RESAMPLES)
    return point, run


def _class_count_bootstrap(data: _Data) -> Tuple[float, Callable[[], Dict[str, Any]]]:
    """M23 — 정답 클래스 수를 다항 재표집한 다수/소수 비. 클래스가 빠지면 비는 무한대."""
    true_col = data.m.get("y_true") or data.m.get("true_labels")
    y_true = data.df[true_col]
    if not y_true.empty and isinstance(y_true.iloc[0], list):
        counts = pd.Series([label for cell in y_true for label in cell]).value_counts()
    else:
        counts = y_true.value_counts()
    counts = counts.to_numpy(dtype=float)
    total = int(counts.sum())
    point = float(counts.max() / counts.min())

    def run() -> Dict[str, Any]:
        rng = np.random.default_rng(SEED)
        draws = rng.multinomial(total, counts / total, size=N_RESAMPLES).astype(float)
        mins = draws.min(1)
        ratios = np.full(N_RESAMPLES, np.inf)
        np.divide(draws.max(1), mins, out=ratios, where=mins > 0)
        return _percentile(ratios, "bootstrap_multinomial", N_RESAMPLES)
    return point, run


# ── 지표 → 방식 ────────────────────────────────────────────────────────

_WILSON_BY_TASK = {
    "binary": {"M1", "M2", "M3", "M7", "M8"},
    "multiclass": {"M1", "M12"},
    "multilabel": {"M1", "M16"},
}
_CONFUSION_BY_TASK = {
    "binary": {"M4", "M5", "M6", "M20"},
    "multiclass": {"M2", "M3", "M4", "M5", "M6", "M11", "M13", "M14", "M20"},
}
_ML_ROW_METRICS = {"M2", "M3", "M4", "M5", "M15", "M17", "M18"}

_NO_ANALYTIC = "표본이 커서 재표집 작업량 상한을 넘었고 이 지표에는 해석적 구간이 없음"


def _wilson_counts(metric_id: str, task: str, data: _Data) -> Tuple[int, int]:
    """Wilson 의 (성공 수, 분모) — 원 지표와 같은 양성 기준·같은 이진화."""
    if task == "multilabel":
        Yt, Yp = data.ml_true_fit() if metric_id == "M1" else data.ml_union_fit()
        return int((Yt == Yp).all(1).sum()), len(Yt)

    tp, t, p, n = _confusion_counts(data)
    if metric_id in ("M1", "M12"):
        return int(tp.sum()), n

    y_true, _ = common._get_true_pred(data.df, data.m)
    if metric_id in ("M2", "M3"):
        k = data.label_index(common._get_binary_kwargs(y_true, data.m)["pos_label"])
    else:
        k = data.label_index(binary._resolve_positive_value(y_true, data.m.get("_positive_class")))
    tp_k, t_k, p_k = (0, 0, 0) if k is None else (int(tp[k]), int(t[k]), int(p[k]))
    tn, fp = n - t_k - p_k + tp_k, p_k - tp_k
    return {
        "M2": (tp_k, p_k),
        "M3": (tp_k, t_k),
        "M7": (tn, tn + fp),
        "M8": (fp, tn + fp),
    }[metric_id]


def _confusion_stat(metric_id: str, task: str, data: _Data) -> Callable:
    beta = float(data.m.get("_beta", 1.0) or 1.0)
    if metric_id == "M6":
        return _kl_stat
    if metric_id == "M20":
        return _mcc_stat
    if task == "binary":
        y_true, _ = common._get_true_pred(data.df, data.m)
        k = data.label_index(common._get_binary_kwargs(y_true, data.m)["pos_label"])
        return _binary_fbeta_stat(k, 1.0 if metric_id == "M4" else beta)
    return {
        "M2": _macro_stat("precision"),
        "M3": _macro_stat("recall"),
        "M4": _macro_stat("fbeta", 1.0),
        "M5": _macro_stat("fbeta", beta),
        "M11": _m11_stat,
        "M13": _m13_stat,
        "M14": _tvd_stat,
    }[metric_id]


def _check_point(point: float, value: float) -> None:
    if not (math.isfinite(point) and abs(point - value) <= POINT_TOLERANCE):
        raise _Skip("점추정 불일치")


def _score_interval(metric_id: str, data: _Data, value: float) -> Dict[str, Any]:
    scores, y = data.binary_scores()
    n = len(scores)
    if metric_id == "M19":
        rows = _log_loss_rows(scores, y)
        point, run = _row_bootstrap(n, 1, _mean_stat(rows))
        _check_point(point, value)
        return run() if run else _normal_interval(rows)

    stat = _auroc_stat(scores, y) if metric_id == "M9" else _auprc_stat(scores, y)
    point, run = _row_bootstrap(n, 1, stat)
    _check_point(point, value)
    if run:
        return run()
    if metric_id == "M9":
        auc, interval = _delong(scores, y)
        _check_point(auc, value)
        return interval
    return _logit_ap(value, int(y.sum()))


def _multilabel_interval(metric_id: str, data: _Data, value: float) -> Dict[str, Any]:
    beta = float(data.m.get("_beta", 1.0) or 1.0)
    if metric_id in ("M2", "M3", "M4", "M5"):
        Yt, Yp = data.ml_true_fit()
        kind = {"M2": "precision", "M3": "recall"}.get(metric_id, "fbeta")
        stat = _ml_macro_stat(Yt, Yp, kind, 1.0 if metric_id == "M4" else beta)
        analytic = None
    else:
        Yt, Yp = data.ml_union_fit()
        if metric_id == "M15":
            stat = _hamming_stat(Yt, Yp)
            analytic = lambda: _normal_interval((Yt != Yp).sum(1) / Yt.shape[1], upper=1.0)
        elif metric_id == "M17":
            rows = _jaccard_rows(Yt, Yp)
            stat = _mean_stat(rows)
            analytic = lambda: _normal_interval(rows, upper=1.0)
        else:  # M18
            stat = _ml_cosine_stat(Yt, Yp)
            analytic = None
    point, run = _row_bootstrap(len(Yt), Yt.shape[1], stat)
    _check_point(point, value)
    if run:
        return run()
    if analytic is None:
        raise _Skip(_NO_ANALYTIC)
    return analytic()


def _interval_for(metric_id: str, task: str, data: _Data, value: float) -> Dict[str, Any]:
    if metric_id in _WILSON_BY_TASK.get(task, ()):
        k, n = _wilson_counts(metric_id, task, data)
        if n > 0:
            _check_point(k / n, value)
        low, high = wilson_interval(k, n)
        return _interval(low, high, "wilson")

    if metric_id == "M23":
        point, run = _class_count_bootstrap(data)
        _check_point(point, value)
        return run()

    if metric_id in _CONFUSION_BY_TASK.get(task, ()):
        point, run = _multinomial_bootstrap(data, _confusion_stat(metric_id, task, data))
        _check_point(point, value)
        return run()

    if task == "binary" and metric_id in ("M9", "M10", "M19"):
        return _score_interval(metric_id, data, value)

    if task == "multilabel" and metric_id in _ML_ROW_METRICS:
        return _multilabel_interval(metric_id, data, value)

    raise _Skip("이 지표의 구간 방식이 정해지지 않음")


def _metric_value(raw: Any) -> Optional[float]:
    """엔진 지표 값 → 구간 대상 수치. 실패·값 없음이면 None. 다중값 지표(M11~M13)는 f1_score."""
    if isinstance(raw, dict):
        if METRIC_ERROR_KEY in raw:
            return None
        raw = raw.get("f1_score")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.floating, np.integer)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def calculate_confidence_intervals(
    df: pd.DataFrame,
    mapping_dict: dict,
    selected_metric_ids: List[str],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """고른 지표마다 95% 신뢰구간. 키 = 지표 ID. 계산 실패한 지표·M21·M22 는 빠진다.

    results 는 engine 이 이미 계산한 지표 값이다 — 점추정 일치 검사의 기준이 된다.
    """
    task = mapping_dict.get("_task_type")
    data = _Data(df, mapping_dict)
    out: Dict[str, Any] = {}
    for metric_id in dict.fromkeys(selected_metric_ids):
        if metric_id in EXCLUDED:
            continue
        value = _metric_value(results.get(metric_id))
        if value is None:
            continue
        try:
            out[metric_id] = _interval_for(metric_id, task, data, value)
        except _Skip as reason:
            out[metric_id] = _skipped(str(reason))
        except Exception as e:  # 한 지표의 구간 실패가 다른 지표·평가 전체를 막지 않게
            logger.warning("신뢰구간 계산 실패 (%s): %s", metric_id, e)
            out[metric_id] = _skipped("구간 계산 중 오류가 나 생략함")
    return out
