"""app/evaluation/errors.py — 지표 계산 실패 표현의 단일 출처.

지표 계산이 실패하면 engine 은 해당 지표 자리에 `{METRIC_ERROR_KEY: "사유"}` 를 넣고,
report 는 그 키로만 실패를 판정한다.

왜 전용 키인가 — 종전에는 평범한 `"error"` 키를 썼는데, M21(혼동행렬)·M22(클래스별
지표)처럼 **dict 를 정상 반환하는 지표**가 같은 네임스페이스를 공유한다. M22 는
classification_report 의 출력이라 클래스명이 그대로 최상위 키가 되므로, 데이터에
'error' 라는 이름의 클래스가 하나라도 있으면 정상 계산된 M22 가 실패로 분류되고
그 지표값 dict 가 실패 사유 문자열 자리에 들어갔다(ISSUES.md C-07).

`__` 접두 이름은 데이터에서 유래한 클래스명과 충돌할 수 없다.

상호작용
- 사용처: app.evaluation.engine(생성), app.evaluation.report(판정)
"""

METRIC_ERROR_KEY = "__metric_error__"


def metric_error(reason: str) -> dict:
    """지표 계산 실패를 나타내는 표준 형태."""
    return {METRIC_ERROR_KEY: reason}
