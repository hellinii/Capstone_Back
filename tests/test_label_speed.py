"""숫자 라벨 평가 시간 회귀 — 10MB 안팎 CSV(정수 0/1 라벨 80만 행)의 이진 전체 지표.

표기 맞추기(S2)가 정수 라벨을 object 문자열로 바꾸자 sklearn 이 지표마다 문자열 배열을 정렬해, 같은 데이터가
옛 엔진(b13e55c) 2.3초 → 42초가 됐다(라운드 C 검토, 업로드 상한 20MiB 안). Render 무료(0.1 CPU)에서는 몇 배 더
느려 요청 타임아웃·재시작이 날 수 있다.

절대 시간은 기계마다 달라 상한으로 쓰지 않는다. 같은 프로세스에서 **정수 배열 sklearn 호출의 시간**(기준)을 재고
평가 시간을 그 배수로 본다 — 느린 CI 에서는 둘이 함께 느려진다. 개발 노트북 실측(기준 = 정수 80만 행
precision_recall_fscore_support 10회 ≈ 0.34초): 옛 엔진 약 7배, 고친 엔진 약 7~8배, 문자열로 바꾸던 엔진 약 120배.
상한 30배는 고친 쪽의 네 배, 느린 쪽의 4분의 1이다.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from app.evaluation.engine import VALID_METRICS_BY_TASK, evaluate

N_ROWS = 800_000
MAX_RATIO = 30


def _reference_seconds(y, p) -> float:
    def once():
        t = time.perf_counter()
        for _ in range(10):
            precision_recall_fscore_support(y, p, average=None)
        return time.perf_counter() - t
    return min(once() for _ in range(3))


def test_integer_labels_evaluate_within_bound_of_sklearn_baseline():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, N_ROWS)
    score = np.clip(y * 0.3 + rng.random(N_ROWS) * 0.7, 0, 1)
    pred = (score >= 0.5).astype(int)
    df = pd.DataFrame({"t": y, "p": pred, "prob_1": score})
    reference = _reference_seconds(y, pred)

    started = time.perf_counter()
    result = evaluate(
        df,
        [{"column": "t", "role": "y_true"}, {"column": "p", "role": "y_pred"},
         {"column": "prob_1", "role": "score_positive"}],
        "binary", sorted(VALID_METRICS_BY_TASK["binary"]),
    )
    elapsed = time.perf_counter() - started

    assert "error" not in result
    assert result["M21"]["labels"] == ["0", "1"]  # 응답의 클래스 이름은 그대로 문자열
    assert elapsed < MAX_RATIO * reference, (
        f"정수 라벨 {N_ROWS}행 평가 {elapsed:.2f}초 — 기준(sklearn 정수 호출 {reference:.3f}초)의 "
        f"{elapsed / reference:.0f}배(상한 {MAX_RATIO}배)"
    )
