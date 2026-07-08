"""evaluation 도메인 — 확정된 매핑으로 ISO/IEC TS 4213 평가지표(TC)를 계산한다.
요청 흐름의 2단계(평가).

파일 목차
- router.py        : 평가 API (POST /api/evaluate) — 얇은 HTTP
- service.py       : 평가 파이프라인 오케스트레이션(충돌검사→계산→리포트 조립)
- engine.py        : 전처리 후 TC 를 계산 함수로 디스패치(METRIC_REGISTRY)
- preprocessor.py  : 계산 전 데이터 정리·검증(단계 헬퍼 파이프라인)
- report.py        : 계산 결과를 성공/실패로 분류·포매팅
- metrics/         : task별 실제 지표 계산(sklearn) — common/binary/multiclass/multilabel
- schemas.py       : 평가 요청/응답 스키마
"""
