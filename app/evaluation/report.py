"""app/evaluation/report.py — 평가 결과 포매팅

engine 이 계산한 원시 결과(지표별 값/에러)를 성공(success_metrics)/실패(failed_metrics)로
분류해 응답용 형태로 정리한다.

상호작용
- 의존(import): 표준 typing 만
- 사용처: app.evaluation.service (EvaluateResponse 조립 직전)
"""

from typing import Dict, Any

from .errors import METRIC_ERROR_KEY

def generate_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    계산된 지표(Metric) 결과들을 모아 최종 API 응답 형태로 포매팅합니다.
    성공한 지표와 실패한(에러가 난) 지표를 별도 오브젝트로 분류하여 반환합니다.

    실패 판정은 engine 이 붙인 전용 키(METRIC_ERROR_KEY)로만 한다. 종전에는 평범한
    "error" 키 존재만 봤는데, M21/M22 처럼 dict 를 정상 반환하는 지표가 같은
    네임스페이스를 써서 데이터에 'error' 라는 이름의 클래스가 있으면 정상 계산된
    M22 가 실패로 분류됐다(ISSUES.md C-07).
    """
    success_metrics = {}
    failed_metrics = {}

    for metric_id, val in results.items():
        # 전처리 메타데이터 등 특수 키는 보존
        if metric_id.startswith("_"):
            continue

        if isinstance(val, dict) and METRIC_ERROR_KEY in val:
            failed_metrics[metric_id] = val[METRIC_ERROR_KEY]
        else:
            success_metrics[metric_id] = val
            
    return {
        "success_metrics": success_metrics,
        "failed_metrics": failed_metrics
    }
