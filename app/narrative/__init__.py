"""narrative 도메인 — 평가 결과를 LLM 이 자연어 서술로 설명한다(성적서 7·8·9절).
요청 흐름의 3단계(서술). 숫자는 grounding 으로 환각을 방어한다.

파일 목차
- router.py         : 서술 API — 옛 POST /api/generate-narrative, 작업 번호 POST·GET /api/narrative/jobs
- jobs.py           : 작업 번호 방식(narrative_jobs 표·run_job·유실 판정·입력 상한 줄이기)
- generator.py      : 작업 경로 생성기 — 장별 호출 + 칸별 검문 사다리(서술 설계 D36)
- prompt.py         : 작업 경로 장별 지시문(한국어)·칸 스키마, TASK_TYPE_LABELS
- fact_view.py      : AI 가 읽는 사실 시트 보기(한국어 이름·코드값 없음)
- checks.py         : 칸별 검사(숫자·판정 모순·예측 수치·D24·형식)
- legacy.py         : 옛 경로의 한 번 호출 생성(S1 판 그대로 얼림 — 다음 라운드에 지움)
- llm.py            : Claude 호출 한 곳(대체 모델·구조화 출력·호출 기록)
- required_items.py : 필수 서술 항목 목록(5절)과 결론 목록(D7)
- sections.py       : 소절 7칸 등록표(칸·장·응답 필드)
- grounding.py      : 환각 방어 — 숫자 허용 목록·판정 모순 검사
- derived.py        : 서버 파생 계산(혼동행렬 합계·분포 백분율)
- fallback.py       : 규칙 템플릿(칸별 함수 7개)
- baselines.py      : 지표별 내부 참조 기준치 테이블
- schemas.py        : FactSheet·서술 요청/응답·출력 스키마
- models.py         : narrative_jobs 표
"""
