"""test_p4_bc_strong.py — 판 4 라운드 S1 ④: RPT-2026-0001 원본 입력(bc_strong)의 평가·검증 정답과 독립 재계산.

- 입력 `tests/data/bc_strong/`: 판 3 발급본 RPT-2026-0001 을 만든 CSV(바이트 그대로, 3,727바이트)와 그때 앱이 보낸 평가 요청
  (payload). 판 4 성적서 대조(계획 6.1)가 이 입력과 아래 정답을 쓴다.
- 정답 `tests/golden_ed4/`(정적 파일, 갱신 경로 없음 — 파일이 없으면 실패한다):
  - `evaluate_bc_strong.json`: 지금 `/api/evaluate` 응답. 비결정 값(평가 시각·라이브러리 버전)은 자리표시자.
  - `validate_bc_strong.json`: 지금 `/api/validate-data` 응답(운영 main 으로 뜬 프론트 입력 사본과 같았다).
  - `evaluate_bc_strong.edition3_base.json`: 지난 라운드(판 3 발급 때) 실제 평가 응답 원문 — "옛 키·값 ⊆ 새 응답" 기준
    (test_p4_evaluate_additive.py).
- 비교는 글자 그대로에 가깝게 한다: 키 집합·문자열·정수·불리언은 정확히, 실수는 소수 12자리(말단 비트만 흡수 —
  기존 골든의 6자리보다 훨씬 좁다).
- 독립 재계산은 엔진 코드를 쓰지 않고 pandas·sklearn·Wilson 식으로 다시 센다.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, log_loss, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from app.main import app
from golden_utils import _redact

DATA = Path(__file__).parent / "data" / "bc_strong"
GOLDEN = Path(__file__).parent / "golden_ed4"
CSV = DATA / "bc_strong.csv"
PAYLOAD = DATA / "bc_strong.payload.json"

# RPT-2026-0001 원본 CSV 의 지문(라운드 기록·계획 1절). 판 4 2.2 '파일 해시'·'파일 크기'에 찍힐 값.
CSV_SHA256 = "4ead9f2ccc8da32c0c3148f02ea6ff4e030c156d9873eeb99883df73f3a49f9d"
CSV_SIZE = 3727

client = TestClient(app)


def _canon(o):
    if isinstance(o, bool) or o is None or isinstance(o, (str, int)):
        return o
    if isinstance(o, float):
        return ("float", round(o, 12))  # 1 과 1.0 을 가른다(직렬화가 다르다)
    if isinstance(o, dict):
        return {k: _canon(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_canon(v) for v in o]
    raise TypeError(type(o))


def _read_golden(name: str):
    path = GOLDEN / name
    assert path.is_file(), f"정답 파일이 없다(만들지 않는다): {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _post(route: str):
    return client.post(
        route,
        files={"file": ("bc_strong.csv", CSV.read_bytes(), "text/csv")},
        data={"data": PAYLOAD.read_text(encoding="utf-8")},
    )


@pytest.fixture(scope="module")
def evaluated():
    resp = _post("/api/evaluate")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_input_bytes_are_the_issued_report_source():
    content = CSV.read_bytes()
    assert len(content) == CSV_SIZE
    assert hashlib.sha256(content).hexdigest() == CSV_SHA256
    assert b"\r" not in content  # 줄 끝이 바뀌면(체크아웃 설정) 지문이 달라진다


def test_evaluate_equals_static_golden(evaluated):
    assert _canon(_redact(evaluated)) == _canon(_read_golden("evaluate_bc_strong.json"))


def test_validate_equals_static_golden():
    resp = _post("/api/validate-data")
    assert resp.status_code == 200, resp.text
    assert _canon(resp.json()) == _canon(_read_golden("validate_bc_strong.json"))


def test_canon_has_teeth():
    assert _canon({"a": 0.9496962348678732}) == _canon({"a": 0.9496962348678731})  # 말단 비트만 흡수
    assert _canon({"a": 0.9496962348678732}) != _canon({"a": 0.949696234867})
    assert _canon({"a": 1}) != _canon({"a": 1.0})
    assert _canon({"a": 1}) != _canon({"a": 1, "b": None})
    assert _canon({"a": [1, "x"]}) != _canon({"a": [1, "y"]})


# ── 독립 재계산(엔진을 쓰지 않는다) ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def frame():
    df = pd.read_csv(CSV)
    y = (df["diagnosis"] == "malignant").astype(int).to_numpy()
    s = df["score_malignant"].astype(float).to_numpy()
    return y, s


def _wilson(k: int, n: int, z: float = 1.959963984540054):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def test_trace_fields(evaluated):
    assert evaluated["source_file"] == {"sha256": CSV_SHA256, "size_bytes": CSV_SIZE}
    assert evaluated["engine_version"] == "1.0.0"


def test_metrics_match_sklearn(frame, evaluated):
    y, s = frame
    pred = (s >= 0.5).astype(int)
    sm = evaluated["results"]["success_metrics"]
    assert sm["M1"] == pytest.approx(accuracy_score(y, pred), abs=1e-12)
    assert sm["M2"] == pytest.approx(precision_score(y, pred), abs=1e-12)
    assert sm["M3"] == pytest.approx(recall_score(y, pred), abs=1e-12)
    assert sm["M4"] == pytest.approx(f1_score(y, pred), abs=1e-12)
    assert sm["M9"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert sm["M10"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert sm["M19"] == pytest.approx(log_loss(y, s), abs=1e-12)
    # 불균형 비율 = 다수/소수 = 107/64 = 1.671875(판 4 는 1.672 로 찍는다).
    assert sm["M23"] == 107 / 64 == 1.671875
    assert sm["imbalance_effect"]["imbalance_ratio_raw"] == 1.671875
    assert sm["confusion_counts"] == {"fn": 1, "fp": 2, "positive_class": "malignant", "tn": 105, "tp": 63, "type": "binary"}


def test_f1_max_threshold_matches_sklearn(frame, evaluated):
    """F1 최대 임계값 0.6166, 그 점 R·P 0.984375, 지금 F1 0.976744(상승폭 0.0076) — 계획 1절 실데이터 확인값."""
    y, s = frame
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-300), 0.0)
    best = int(np.argmax(np.round(f1, 9)))
    gain = evaluated["results"]["success_metrics"]["f1_threshold_gain"]

    assert thresholds[best] == 0.6166 == gain["best_threshold_raw"]
    assert gain["best_recall_raw"] == recall[best] == 0.984375
    assert gain["best_precision_raw"] == precision[best] == 0.984375
    assert gain["max_f1_raw"] == pytest.approx(0.984375, abs=1e-12)
    pred = (s >= 0.5).astype(int)
    assert gain["current_recall_raw"] == pytest.approx(recall_score(y, pred), abs=1e-12)
    assert gain["current_precision_raw"] == pytest.approx(precision_score(y, pred), abs=1e-12)
    assert gain["current_f1_raw"] == pytest.approx(f1_score(y, pred), abs=1e-12)
    assert round(gain["current_f1_raw"], 6) == 0.976744
    assert round(gain["gain_raw"], 6) == 0.007631

    # 표기: 0.617 을 실제로 적용하면 0.6166 인 표본이 음성이 되어 Recall 이 0.984 → 0.969 로 바뀐다 → "0.6166".
    assert round(recall_score(y, (s >= 0.617).astype(int)), 3) == 0.969
    assert recall_score(y, (s >= 0.6166).astype(int)) == 0.984375
    assert gain["best_threshold_text"] == "0.6166"
    # 기존 칸(판 3 이 쓰는 넷째 자리 값)은 그대로.
    assert {k: gain[k] for k in ("max_f1", "current_f1", "gain", "best_threshold", "notable", "gain_text")} == {
        "max_f1": 0.9844, "current_f1": 0.9767, "gain": 0.0076, "best_threshold": 0.6166,
        "notable": False, "gain_text": "0.0076",
    }


def test_accuracy_interval_matches_independent_wilson(frame, evaluated):
    """M1 Wilson 구간 [0.9497, 0.994] — 반올림 전 하한 0.94969… 이 목표 0.950 과 같아 보이는 판 4 예(계획 2.7)."""
    y, s = frame
    pred = (s >= 0.5).astype(int)
    low, high = _wilson(int((pred == y).sum()), len(y))
    ci = evaluated["results"]["success_metrics"]["confidence_intervals"]["M1"]
    assert ci["low_raw"] == pytest.approx(low, abs=1e-12)
    assert ci["high_raw"] == pytest.approx(high, abs=1e-12)
    assert (ci["low"], ci["high"]) == (0.9497, 0.994)
    assert ci["low_raw"] < 0.95 and round(ci["low_raw"], 3) == 0.95
