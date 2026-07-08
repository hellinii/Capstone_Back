"""analysis 도메인 — 업로드 파일을 받아 컬럼 역할을 파악하고 매핑을 검증한다.
요청 흐름의 1단계(분석). 두 하위 흐름: [컬럼 매핑]과 [데이터 검증].

파일 목차
- router.py             : 컬럼 매핑 API (POST /api/analyze-columns, /api/confirm-mapping) — 얇은 HTTP
- validation_router.py  : 데이터 검증 API (POST /api/validate-data) — 얇은 HTTP
- analysis_service.py   : 컬럼 매핑 전략 오케스트레이션(무키→규칙폴백 / LLM 실패→강등)
- validation_service.py : 데이터 검증 파이프라인 오케스트레이션 + 응답 조립
- llm_mapper.py         : LLM 으로 컬럼 자동 매핑(외부 경계)
- fallback_mapper.py    : 규칙 기반 컬럼 매핑 폴백
- reconcile.py          : LLM 반환 컬럼명을 실제 파일 헤더에 정렬(신뢰 경계)
- metadata.py           : 확정 매핑 기반 메타데이터(클래스/분포) 추출
- prompt_builder.py     : 컬럼 매핑용 LLM 프롬프트 생성
- validator.py          : 매핑 유효성 검사 + 계산 가능 지표 산출, 컬럼 충돌 탐지
- validation_checks.py  : 데이터 검증 개별 점검 함수 모음
- schemas.py            : 분석 도메인 요청/응답 스키마
"""
