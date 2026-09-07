# Capstone 프로젝트 개요 (Start Here)

> **한 줄 소개**: KS X ISO/IEC TS 4213:2022 표준 기반 **AI 분류 모델 성능 시험성적서 자동 생성 서비스**.
> 사용자가 모델의 예측 결과 파일(CSV/JSON)을 올리면 → 컬럼 자동 매핑 → 표준 지표(M1~M23) 계산 → LLM 서술 보강 → 공식 성적서 발급까지 수행한다. (모델을 실행하는 게 아니라 **이미 나온 예측 결과를 채점**한다.)

이 문서는 새 세션/새 개발자가 **프론트+백 전체 구조를 5분 안에 파악**하도록 돕는 진입 지도다. 세부는 각 문서로 연결만 하고 중복 서술하지 않는다.

> **위치**: 이 파일은 `Capstone_Back/docs/PROJECT_OVERVIEW.md` 다(2026-09-07 ★결정 11 로 이동).
> 종전에는 두 저장소의 **부모 폴더**에 있었는데, 그 폴더는 git 저장소가 아니라 로컬 전용이라
> **아무도 이 문서를 받아볼 수 없었다** — 저장소를 클론한 사람에게는 존재하지 않는 문서였다.
> 프론트 내용도 함께 다루지만 두 저장소 중 하나를 골라야 하므로, 스택·배포 설명이 더 무거운
> 백엔드에 둔다.

---

## 1. 저장소 · 스택 · 배포

| | Capstone_Front | Capstone_Back |
|---|---|---|
| 역할 | UI / 7단계 워크플로우 / 성적서 렌더 | 컬럼 매핑·검증·지표계산·LLM 서술·발급 API |
| 스택 | React 18 · TS · Vite · Zustand · Tailwind/shadcn · pnpm | Python 3.12 · FastAPI · pandas · scikit-learn · OpenAI · SQLAlchemy · Pydantic |
| 배포 | **Vercel** | **Render** web service + **Neon PostgreSQL** (로컬은 SQLite) |
| 원격 | github.com/hwanginyong02/Capstone_Front | github.com/hwanginyong02/Capstone_Back |

- **task_type 3종**: `binary` / `multiclass` / `multilabel`.
- **지표 식별자**: **M1~M23** (SPEC·프론트·API 공통). 백엔드 내부 변수명도 `METRIC_REQUIREMENTS` 로 통일돼 옛 `TC` 표기는 남아 있지 않다(2026-09-07 확인).
- **LLM**: OpenAI `gpt-4.1-nano`를 두 곳에 사용 — ①컬럼 자동 매핑 ②성적서 서술. **키가 없거나 실패해도 규칙 기반 폴백**으로 서비스가 계속 동작한다.
- ⭐ **단일 진실 문서(SSoT)**: `Capstone_Front/SPEC.md` — 지표별 필수/선택 컬럼·컬럼 역할·task_type별 허용 지표·검증 규칙. **프론트/백 양쪽이 이 문서 기준으로 구현**한다.

---

## 2. 디렉토리 지도

```
/Users/minseokim/Capstone
├── PROJECT_OVERVIEW.md         # (이 문서) 프론트+백 통합 진입 지도
├── Capstone_Front/             # 프론트엔드
└── Capstone_Back/              # 백엔드
```

### Capstone_Front — "페이지 = 얇은 조립자(Assembler)" 아키텍처
```
Capstone_Front
├── SPEC.md                     # ⭐ 지표/컬럼/검증 단일 진실 문서 (프론트+백 공통)
├── docs/ARCHITECTURE.md        # 프론트 구조 기준 문서
├── docs/FRONTEND_DEVELOPMENT_GUIDELINE.md   # 개발 시 지켜야 할 구조 규칙
└── src/
    ├── App.tsx / routes.ts      # 라우팅 (7단계 워크플로우는 /app/*)
    ├── pages/                   # 얇은 조립자 (단계별 페이지 + report/ + workspaces/)
    ├── components/              # 도메인별 UI (basic-info, ..., report, workspaces, landing, ui)
    ├── layout/                  # WorkflowShell(워크플로우 셸) · AppShell(비-워크플로우 셸)
    ├── hooks/                   # useColumnAnalysis · useDataValidation · useReportData · useIssuance · usePrintOnReady ...
    ├── lib/                     # apiBase(apiUrl) · mapping(role 변환) · report(API·factSheet·평가로직)
    ├── utils/                   # styling · format · domain · stores(Zustand: useWorkflowStore/useWorkspaceStore)
    └── data/ · types/ · styles/ · test/
```
7단계 워크플로우: **기본정보 → 지표선택 → 지표상세 → 데이터업로드 → 컬럼매핑 → 데이터검증 → 리포트**.

### Capstone_Back — "라우터(얇은 HTTP) → 서비스(오케스트레이션) → 순수 로직" 계층
```
Capstone_Back
├── README.md                   # 2줄 stub (온보딩·구조는 docs/ARCHITECTURE.md 참조)
├── docs/                       # ARCHITECTURE · DEPLOYMENT_PLAN · ISSUANCE_DB_DESIGN · REPORT_NARRATIVE_DESIGN
├── render.yaml · requirements.txt · pytest.ini · tests/
└── app/
    ├── main.py                  # FastAPI 진입점 (lifespan: init_db+seed / CORS / 5개 라우터 / /health)
    ├── core/                    # schemas(enum·역할규칙·지표요건표) · parsing(CSV/JSON→DataFrame) · database
    ├── analysis/                # 컬럼 매핑 + 데이터 검증 (llm_mapper / fallback_mapper / validator / metadata)
    ├── evaluation/              # ISO4213 지표 계산 (engine · metrics/{common,binary,multiclass,multilabel})
    ├── narrative/               # LLM 서술(7·8·9절) (grounding 환각방어 · fallback · baselines)
    └── issuance/                # 성적서 발급/채번 DB 도메인 (models · service · serializers)
```

---

## 3. 핵심 데이터 흐름 (워크플로우 → API → 백엔드 → 성적서)

```
[Step4 데이터업로드]  파일 + task_type
   └ POST /api/analyze-columns → analysis_service (LLM 매핑, 실패/무키 시 규칙 폴백)
        → 프론트 translateRoleToFrontend 로 role 역변환

[Step5 컬럼매핑]  사용자 매핑 확정
   └ POST /api/confirm-mapping → validator.validate_mapping → is_valid + available_metric_ids

[Step6 데이터검증]  전처리 dry-run (지표 계산 없음)
   └ POST /api/validate-data → validation_service → validation_details / execution_summary
        → store.setValidationResult 보존 (리포트 6절 재사용)

[리포트 Stage1 — 결정론적 KPI·차트]
   └ POST /api/evaluate → evaluation.service.run_evaluation_pipeline
        → results(success/failed_metrics, M21 혼동행렬/M22 리포트/M23 불균형비, roc/pr_curve, latency)

[리포트 Stage2 — LLM 서술]
   └ POST /api/generate-narrative → narrative.service (파생값 → 화이트리스트 → LLM → grounding → 폴백)
        ※ verdict/score 는 프론트 computeVerdict 가 권위, 백엔드는 fact_sheet 값으로 강제

[성적서 발급/조회]
   └ POST /api/reports/issue (run_id 멱등) · reissue(번호 유지·버전업) · GET 조회
        ※ isEvaluated 인 run 에만 발급 허용
```

---

## 4. 프론트 ↔ 백 API 계약

모든 URL은 프론트 `src/lib/apiBase.ts` 의 `apiUrl()`(`VITE_API_BASE_URL`+path, 미설정 시 상대경로→Vite 프록시)로 생성. 백엔드 라우터 prefix는 모두 `/api`(`/health` 제외).

| HTTP / 경로 | 프론트 호출부 | 백엔드 핸들러 → 위임 | 단계 |
|---|---|---|---|
| POST `/api/analyze-columns` | `hooks/useColumnAnalysis.ts` | `analysis/router.py` → `resolve_column_mapping` | Step4 |
| POST `/api/confirm-mapping` | `lib/report/confirmMappingApi.ts` | `analysis/router.py` → `validate_mapping` | Step5 |
| POST `/api/validate-data` | `hooks/useDataValidation.ts` | `analysis/validation_router.py` → `validate_dataset` | Step6 |
| POST `/api/evaluate` | `hooks/useReportData.ts` | `evaluation/router.py` → `run_evaluation_pipeline` | 리포트 S1 |
| POST `/api/generate-narrative` | `lib/report/fetchNarrative.ts` (payload=`buildFactSheet.ts`) | `narrative/router.py` → `generate_narrative` | 리포트 S2 |
| GET `/api/organization`, POST `/api/reports/issue`·`{no}/reissue`, GET `/api/reports/{no}` | `lib/report/issuanceApi.ts` · `hooks/useIssuance.ts` | `issuance/router.py` → `service` | 발급 |

- **`/api/validate-data` 와 `/api/evaluate` 는 동일 요청 구조** `EvaluateRequest`(`task_type, column_mappings[], selected_metric_ids[], metadata, beta`), multipart(`file` + `data`=JSON).
- **role 어휘 변환(drift 방지 단일 출처 2함수)**: 요청 조립 `src/lib/mapping/translateRoleToBackend.ts`(task_type 의존), 응답 역변환 `translateRoleToFrontend.ts`. 백엔드 대응물은 `app/core/schemas.py` 의 `ColumnRole` / `VALID_ROLES_BY_TASK` / `METRIC_REQUIREMENTS`(= SPEC.md의 코드 구현).
  - 예: `y_true`→(binary/multiclass) `y_true` · (multilabel) `true_labels`; `score`→`score_positive`(binary); `prob_class_*`→`prob_per_class`(multiclass); `prob_label_*`→`score_per_label`(multilabel); 공통 `id`→`sample_id`, `latency`→`latency`, `ignore`→`ignore`.
- **graceful 처리**: 조회(GET) 실패 시 `null`, 발급(POST) 실패 시 백엔드 `detail` throw, 서술 실패/무키 시 프론트 `EMPTY`("생성 예정"). snake_case→camelCase 및 KST 표기는 프론트에서 변환.

---

## 5. 문서 색인 (언제 무엇을 읽나)

| 문서 | 언제 읽나 |
|---|---|
| `Capstone_Front/SPEC.md` ⭐ | **지표 요구사항·컬럼 역할·검증 규칙·task_type별 허용 지표를 다루는 모든 작업의 최우선 기준.** 변경 전 프론트/백 양쪽을 이 문서에 맞춘다. |
| `Capstone_Back/docs/ARCHITECTURE.md` | 백엔드 온보딩·구조 기준 — `app/` 도메인 구조, router→service→순수로직 패턴, 새 지표/API/테이블 추가 위치. (`Capstone_Back/README.md`는 2줄 stub) |
| `Capstone_Back/docs/DEPLOYMENT_PLAN.md` | **배포 작업 시** — Render+Neon+Vercel, `DATABASE_URL`/`VITE_API_BASE_URL`, 콜드스타트. |
| `Capstone_Back/docs/ISSUANCE_DB_DESIGN.md` | **발급/채번/재발급** 로직 수정 시 — 3테이블(organization·report·issuance)·연도별 채번. |
| `Capstone_Back/docs/REPORT_NARRATIVE_DESIGN.md` | **성적서 서술(7·8·9절)** 수정 시 — LLM grounding·verdict/score 규칙·fallback·benchmark. |
| `Capstone_Front/docs/ARCHITECTURE.md` | 프론트 처음 진입 — 페이지=조립자, `src/` 계층, Zustand. |
| `Capstone_Front/docs/FRONTEND_DEVELOPMENT_GUIDELINE.md` | 프론트에 새 페이지/컴포넌트/유틸 추가 시 지킬 규칙. |
| `Capstone_Front/README.md` | UI/스타일 작업 시 (색상·타이포·컴포넌트 패턴)과 7단계 목록. |

> **분업 구조**: 백엔드 설계 3문서(DEPLOYMENT_PLAN·ISSUANCE_DB_DESIGN·REPORT_NARRATIVE_DESIGN)는 "현재 구조는 `docs/ARCHITECTURE.md`, 설계 의도·이력은 각 문서"로 역할이 나뉜다.

---

## 6. 로컬 실행 / 배포

### 백엔드 (`Capstone_Back`)
```bash
pip install -r requirements-dev.txt        # requirements.txt 포함
cp .env.example .env                        # OPENAI_API_KEY(없으면 규칙폴백) / DATABASE_URL(없으면 SQLite data/app.db)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # 또는 python -m app.main
pytest                                       # 테스트 (asyncio_mode=auto, pythonpath=.)
```
첫 부팅 시 `init_db()`+`seed_organization()` 자동. OpenAI 키 없이도 기동됨.

### 프론트 (`Capstone_Front`)
```bash
pnpm install
pnpm dev            # Vite dev (VITE_API_BASE_URL 미설정 시 상대경로→프록시)
pnpm typecheck && pnpm test && pnpm build
```

### 배포
- **백엔드 = Render** web service `capstone-back-59z8` (Python 3.12.13, region singapore, `healthCheckPath:/health`). 실제 URL 은 `https://capstone-back-59z8.onrender.com` — 계획서의 `capstone-back.onrender.com` 이 아니다(이름 선점으로 접미사가 붙었다). `autoDeploy:false` — 배포는 **CI(deploy job)의 Render Deploy Hook 호출이 유일 경로**. `DATABASE_URL`/`OPENAI_API_KEY`는 대시보드 입력(커밋 금지). ⚠️ 실제 Start Command는 대시보드 값 사용(`render.yaml`은 문서 정합용).
- **CI** (`.github/workflows/ci.yml`): PR/push(`dev`,`main`) → test job(`pytest -q` + `/health`·`/api/organization` 스모크). `main` push & 통과 시 deploy job이 포크 미러 푸시 후 Deploy Hook POST.
- **프론트 = Vercel** (`vercel.json`), **DB = Neon PostgreSQL**(`ap-southeast-1`).

---

## 7. 새 세션 추천 읽기 순서
1. 이 문서(PROJECT_OVERVIEW.md)로 전체 지도 파악 →
2. 작업 대상에 따라 §5 색인에서 해당 문서 →
3. 지표/컬럼/검증 관련이면 **반드시 `SPEC.md` 먼저** →
4. 코드 진입: 프론트는 `routes.ts`→`pages/`→`components/`, 백엔드는 `app/main.py`→도메인 `router.py`→`service.py`.
