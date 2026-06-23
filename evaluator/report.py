from typing import Dict, Any

def generate_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    계산된 지표(Metric) 결과들을 모아 최종 API 응답 형태로 포매팅합니다.
    성공한 지표와 실패한(에러가 난) 지표를 별도 오브젝트로 분류하여 반환합니다.
    """
    success_metrics = {}
    failed_metrics = {}
    
    for tc_id, val in results.items():
        # 전처리 메타데이터 등 특수 키는 보존
        if tc_id.startswith("_"):
            continue
            
        if isinstance(val, dict) and "error" in val:
            failed_metrics[tc_id] = val["error"]
        else:
            success_metrics[tc_id] = val
            
    return {
        "success_metrics": success_metrics,
        "failed_metrics": failed_metrics
    }
