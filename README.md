# Capstone_Back

KS X ISO/IEC TS 4213:2022 기반 AI 분류 모델 성능 시험성적서 자동 생성 서비스의 **백엔드**(FastAPI). 업로드된 예측 결과 파일을 채점해 표준 지표(M1~M23)를 계산하고, LLM 서술을 보강하며, 성적서 발급/채번을 담당합니다.

> - 코드 구조·설계 규칙: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> - 지표·컬럼·검증 규칙(프론트/백 공통 단일 진실 문서): `../Capstone_Front/SPEC.md`
> - 프로젝트 전체(프론트+백) 지도: `../PROJECT_OVERVIEW.md`
> - 배포/발급/서술 설계: `docs/DEPLOYMENT_PLAN.md` · `docs/ISSUANCE_DB_DESIGN.md` · `docs/REPORT_NARRATIVE_DESIGN.md`

## 기술 스택
Python 3.12 · FastAPI · pandas · scikit-learn · OpenAI(`gpt-4.1-nano`) · SQLAlchemy(로컬 SQLite / 배포 Neon PostgreSQL) · Pydantic

## 구조 (요약, 자세히는 docs/ARCHITECTURE.md)
`app/` 은 도메인별로 나뉘며, 각 도메인은 **라우터(얇은 HTTP) → 서비스(오케스트레이션) → 순수 로직** 계층을 따릅니다.

```
app/
├── main.py       # FastAPI 진입점 (lifespan, CORS, 라우터 등록, /health)
├── core/         # 공용 schemas(enum·역할규칙·지표요건표)·parsing·database
├── analysis/     # 컬럼 자동 매핑(LLM/규칙 폴백) + 데이터 검증
├── evaluation/   # ISO4213 지표 계산 (engine · metrics/{common,binary,multiclass,multilabel})
├── narrative/    # LLM 서술 생성(7·8·9절) + grounding(환각 차단) + 폴백
└── issuance/     # 성적서 발급/채번 (DB 사용)
```

## 주요 API (prefix `/api`)
`POST /api/analyze-columns` · `POST /api/confirm-mapping` · `POST /api/validate-data` · `POST /api/evaluate` · `POST /api/generate-narrative` · `GET/PUT /api/organization` · `POST /api/reports/issue`·`{no}/reissue` · `GET /api/reports/{no}` · `GET /health`

## 로컬 실행
```bash
pip install -r requirements-dev.txt   # requirements.txt 포함
cp .env.example .env                  # OPENAI_API_KEY(없으면 규칙 폴백) / DATABASE_URL(없으면 SQLite data/app.db)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # 또는 python -m app.main
pytest                                # 테스트 (pytest.ini: asyncio_mode=auto)
```
첫 부팅 시 `init_db()` + `seed_organization()` 이 자동 실행됩니다. OpenAI 키 없이도 기동되며 LLM 기능은 규칙 기반으로 폴백합니다.

## 배포
Render web service(`capstone-back`) + Neon PostgreSQL. `autoDeploy:false` — 배포는 CI(`.github/workflows/ci.yml`)의 Render Deploy Hook 호출로 이루어집니다. 자세히는 `docs/DEPLOYMENT_PLAN.md`.

## 라이선스
© 2026 서울과학기술대학교
