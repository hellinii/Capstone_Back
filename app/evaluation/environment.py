"""app/evaluation/environment.py — 평가 수행 환경의 실측 보고.

ISSUES.md F-09 (2026-09-07 ★확정된 제품 결정 4 — 앞으로 발급되는 것만 정정).

성적서 4절에 인쇄되는 '평가 도구'와 '평가 일시'가 프론트 상수에 하드코딩돼 있었다
(`scikit-learn 1.4.0` 등). 그 값은 실제로 계산에 쓰인 버전과 아무 관계가 없었다 —
버전을 아는 곳은 계산을 수행한 프로세스뿐이다.

**이미 발급된 성적서는 손대지 않는다**(결정 4). 서버 스냅샷의 값은 그대로 두고,
앞으로 발급되는 것부터 실측값이 실린다. 그래서 응답 필드는 옵셔널이고 프론트는 구
스냅샷에 대해 종전 상수로 폴백한다.

상호작용
- 의존(import): sys, datetime, numpy, pandas, sklearn
- 사용처: app.evaluation.service (EvaluateResponse 조립)
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import sklearn

# 성적서 발급일과 같은 시간대를 쓴다 — 채번 연도(UTC)와 표기 발급일(KST)이 연 경계에서
# 어긋나던 문제(F-07)와 같은 계열의 실수를 반복하지 않기 위해서다.
KST = timezone(timedelta(hours=9))


def library_versions() -> dict[str, str]:
    """지표 계산에 실제로 쓰인 라이브러리 버전."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def evaluated_at() -> str:
    """평가 수행 시각(KST, ISO 8601)."""
    return datetime.now(KST).isoformat(timespec="seconds")
