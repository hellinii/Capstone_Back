# Capstone_Back

KS X ISO/IEC TS 4213:2022 기반 AI 분류 모델 성능 시험성적서 자동 생성 서비스의 **백엔드**(FastAPI). 업로드된 예측 결과 파일을 채점해 표준 지표(M1~M23)를 계산하고, LLM 서술을 보강하며, 성적서 발급/채번을 담당합니다.

> - 코드 구조·설계 규칙: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> - 지표·컬럼·검증 규칙(프론트/백 공통 단일 진실 문서): 프론트 저장소(Capstone_Front)의 `SPEC.md` — https://github.com/hwanginyong02/Capstone_Front/blob/main/SPEC.md
> - 프로젝트 전체(프론트+백) 지도: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
> - 서술·발급·배포의 자세한 규칙과 이유: `docs/ARCHITECTURE.md` §10(서술) · §11(발급) · §12(배포·운영)

## 기술 스택
Python 3.12 · FastAPI · pandas · scikit-learn · Anthropic(서술 `claude-sonnet-5-5`) · OpenAI(컬럼 매핑 `gpt-6-luna`) · SQLAlchemy(로컬 SQLite / 배포 Neon PostgreSQL) · Pydantic

## 구조 (요약, 자세히는 docs/ARCHITECTURE.md)
`app/` 은 도메인별로 나뉘며, 각 도메인은 **라우터(얇은 HTTP) → 서비스(오케스트레이션) → 순수 로직** 계층을 따릅니다.

```
app/
├── main.py       # FastAPI 진입점 (lifespan, CORS, 라우터 등록, /health)
├── core/         # 공용 schemas(enum·역할규칙·지표요건표)·parsing·database
├── analysis/     # 컬럼 자동 매핑(LLM/규칙 폴백) + 데이터 검증
├── evaluation/   # ISO4213 지표 계산 (engine · metrics/{common,binary,multiclass,multilabel})
├── narrative/    # LLM 서술(7·8·9절) — 작업 번호(job)·장별 생성·칸별 검문 사다리·규칙 템플릿(ARCHITECTURE §10)
└── issuance/     # 성적서 발급/채번 (DB 사용)
```

## 주요 API (prefix `/api`)
`POST /api/analyze-columns` · `POST /api/confirm-mapping` · `POST /api/validate-data` · `POST /api/evaluate` · `POST /api/narrative/jobs`·`GET /api/narrative/jobs/{id}` · `POST /api/generate-narrative`(옛 경로) · `GET /api/organization` · `POST /api/reports/issue`·`{no}/reissue` · `GET /api/reports/{no}` · `GET /api/reports/{report_no}/content`(발급 시점 원본) · `GET /health`

## 로컬 실행
```bash
pip install -r requirements-dev.txt   # requirements.txt 포함
cp .env.example .env                  # ANTHROPIC_API_KEY(서술 — 없으면 규칙 문장) / OPENAI_API_KEY(매핑 — 없으면 규칙 매핑) / DATABASE_URL(없으면 SQLite data/app.db)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # 또는 python -m app.main
venv/bin/python -m pytest            # 테스트 (pytest.ini: asyncio_mode=auto)
```
첫 부팅 시 `init_db()` + `seed_organization()` 이 자동 실행됩니다(발급 표·서술 작업 표, 기관 표가 비어 있으면 기관 행 "리트머스" — 운영 DB 의 기관 이름 바꾸기는 `docs/ARCHITECTURE.md` §12). 키 없이도 기동되며 LLM 기능은 규칙 기반으로 대체됩니다.

## 배포
Render web service(`https://capstone-back-59z8.onrender.com`) + Neon PostgreSQL. 팀 저장소 `main` 에 push 하면 CI(`.github/workflows/ci.yml`)가 테스트·기동 스모크를 돌리고, 통과하면 포크(`hellinii/Capstone_Back`) `main` 으로 복사한 뒤 Render Deploy Hook 을 불러 배포하고 `/health` 의 커밋이 바뀔 때까지 확인합니다. 이것이 유일한 배포 경로입니다. 자세히는 `docs/ARCHITECTURE.md` §12.

## 라이선스
© 2026 Kim Minseo, Hwang Inyong (서울과학기술대학교 졸업과제)
