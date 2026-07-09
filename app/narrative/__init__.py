"""narrative 도메인 — 평가 결과를 LLM 이 자연어 서술로 설명한다(성적서 7·8·9절).
요청 흐름의 3단계(서술). 숫자는 grounding 으로 환각을 방어한다.

파일 목차
- router.py     : 서술 API (POST /api/generate-narrative) — 얇은 HTTP
- service.py    : 서술 생성 오케스트레이션(파생→기준치→LLM→검증→조립, 실패 시 폴백)
- grounding.py  : 환각 방어 — 출력 숫자를 fact_sheet 화이트리스트로 검증
- derived.py    : 서버 파생 계산(혼동행렬·분포·판정 카운트)
- prompt.py     : 서술용 LLM 프롬프트 + 구조화 출력 스키마
- fallback.py   : 규칙 기반 서술 폴백(LLM 실패/검증 위반 시)
- baselines.py  : 지표별 내부 참조 기준치 테이블
- schemas.py    : FactSheet·서술 요청/응답·출력 스키마
"""
