# Capstone 프로젝트 개요 (Start Here)

> **한 줄 소개**: KS X ISO/IEC TS 4213:2022 표준 기반 **AI 분류 모델 성능 시험성적서 자동 생성 서비스**.
> 사용자가 모델의 예측 결과 파일(CSV/JSON)을 올리면 → 컬럼 자동 매핑 → 표준 지표(M1~M23) 계산 → LLM 서술 보강 → 공식 성적서 발급까지 수행한다. (모델을 실행하는 게 아니라 **이미 나온 예측 결과를 채점**한다.)

이 문서는 새 세션/새 개발자가 **프론트+백 전체 구조를 5분 안에 파악**하도록 돕는 진입 지도다. 세부는 각 문서로 연결만 하고 중복 서술하지 않는다.

> 프론트 내용도 함께 다루지만 문서는 한 곳에만 두려고 백엔드 저장소(`Capstone_Back/docs/`)에 둔다.

---

## 1. 저장소 · 스택 · 배포

| | Capstone_Front | Capstone_Back |
|---|---|---|
| 역할 | UI / 7단계 워크플로우 / 성적서 렌더 | 컬럼 매핑·검증·지표계산·LLM 서술·발급 API |
| 스택 | React 18 · TS · Vite · Zustand · Tailwind/shadcn · pnpm | Python 3.12 · FastAPI · pandas · scikit-learn · Anthropic · OpenAI · SQLAlchemy · Pydantic |
| 배포 | **Vercel** | **Render** web service + **Neon PostgreSQL** (로컬은 SQLite) |
| 원격 | github.com/hwanginyong02/Capstone_Front | github.com/hwanginyong02/Capstone_Back |

- **task_type 3종**: `binary` / `multiclass` / `multilabel`.
- **지표 식별자**: **M1~M23** (SPEC·프론트·API 공통). 백엔드 내부 변수명도 `METRIC_REQUIREMENTS` 로 통일돼 옛 `TC` 표기는 남아 있지 않다(2026-09-07 확인).
- **LLM**: 두 곳에 쓴다 — ①컬럼 자동 매핑은 OpenAI(`OPENAI_API_KEY`, `MAPPING_MODEL` 기본 `gpt-6-luna`), ②성적서 서술은 Anthropic Claude Sonnet 5.5(`ANTHROPIC_API_KEY`, `NARRATIVE_MODEL` 기본 `claude-sonnet-5-5`, 작업 경로 effort `NARRATIVE_JOB_EFFORT` 기본 high). 옛 매핑·서술 모델 `gpt-4.1-nano` 는 2026-10-23 에 종료됐다. **키가 없거나 실패해도 규칙 기반**으로 서비스가 계속 동작한다.
- ⭐ **단일 진실 문서(SSoT)**: `Capstone_Front/SPEC.md` — 지표별 필수/선택 컬럼·컬럼 역할·task_type별 허용 지표·검증 규칙. **프론트/백 양쪽이 이 문서 기준으로 구현**한다.

---

## 2. 디렉토리 지도

```
(두 저장소를 나란히 받은 폴더)
├── Capstone_Front/             # 프론트엔드
└── Capstone_Back/              # 백엔드 — 이 문서는 Capstone_Back/docs/PROJECT_OVERVIEW.md
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
├── README.md                   # 스택·구조 요약·주요 API·로컬 실행·배포 (자세한 구조는 docs/ARCHITECTURE.md)
├── docs/                       # ARCHITECTURE(구조·서술·발급·배포) · PROJECT_OVERVIEW(이 문서)
├── render.yaml · requirements.txt · pytest.ini · tests/
└── app/
    ├── main.py                  # FastAPI 진입점 (lifespan: init_db+seed / CORS / 5개 라우터 / /health)
    ├── core/                    # schemas(enum·역할규칙·지표요건표) · parsing(CSV/JSON→DataFrame) · database
    ├── analysis/                # 컬럼 매핑 + 데이터 검증 (llm_mapper / fallback_mapper / validator / metadata)
    ├── evaluation/              # ISO4213 지표 계산 (engine · metrics/{common,binary,multiclass,multilabel})
    ├── narrative/               # LLM 서술(7·8·9절) — jobs(작업 번호) · generator(장별 생성 + 칸별 검문) · required_items(결론 목록) · fallback(규칙 템플릿) · legacy(옛 경로)
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
   └ POST /api/narrative/jobs → 작업 번호 + 8.1 표 즉시 → (뒤에서) narrative.generator: 결론 목록 → 장별 동시 호출 →
        칸별 검사(숫자·판정 모순·권고·선언 숫자·형식) → 걸린 칸만 다시/규칙 문장 → GET /api/narrative/jobs/{id} 로 조회
        ※ 옛 경로 POST /api/generate-narrative(한 번 호출, narrative.legacy)는 옛 클라이언트용으로 얼려 둠 — 운영 프론트는
          2026-10-23 배포부터 작업 경로(src/lib/report/narrativeJobApi.ts)만 쓴다. 다음 라운드에 지운다
        ※ verdict/score 는 프론트 computeVerdict 가 권위, 백엔드는 fact_sheet 값으로 강제
        ※ 서술 작업 경로: 프론트는 작업 번호를 run 에 저장하고 조회만 한다 — done 이어야 완성본(isEvaluated), 그 전에는
          인쇄·PDF·발급 잠금, 실패면 "서술 다시 만들기"(프론트 docs/ARCHITECTURE.md §6.1, 백엔드 §10)

[성적서 발급/조회]
   └ POST /api/reports/issue (run_id 멱등) · reissue(번호 유지·버전업) · GET 조회
        ※ isEvaluated 인 run 에만 발급 허용
```

- **성적서 판(contentSchema)**: 성적서 데이터에 판 번호가 찍힌다(지금 판 3). 발급한 옛 판(1·2)은 번호로 열어도 발급 당시 모양 그대로 그린다. 판 3 은 소절 번호·7.3 추가 관찰·7절 ※ 한 줄이 있고, 본문에 AI·LLM 표현이 없다(1.4 LLM 면책 항목 없음). 프론트 `docs/ARCHITECTURE.md` §6.3.
- **작성 기록**: 칸마다 AI 작성(모델 이름)·규칙 기반 문장·고정 문구인지를 발급 보관본에 저장하고(작업 결과 `section_sources`·`section_models`), 서비스 화면의 "작성 기록" 상자에만 보인다 — 인쇄물에는 없다.
- **기관 이름**: 시드·프론트 자리표시자의 기관 이름은 "리트머스"(옛 "한국 AI 인증원"). 운영 DB 의 기존 행은 SQL 로 한 번 바꿨다(2026-10-23, 방법은 `docs/ARCHITECTURE.md` §12 배포·운영).

---

## 4. 프론트 ↔ 백 API 계약

모든 URL은 프론트 `src/lib/apiBase.ts` 의 `apiUrl()`(`VITE_API_BASE_URL`+path, 미설정 시 상대경로→Vite 프록시)로 생성. 백엔드 라우터 prefix는 모두 `/api`(`/health` 제외).

| HTTP / 경로 | 프론트 호출부 | 백엔드 핸들러 → 위임 | 단계 |
|---|---|---|---|
| POST `/api/analyze-columns` | `hooks/useColumnAnalysis.ts` | `analysis/router.py` → `resolve_column_mapping` | Step4 |
| POST `/api/confirm-mapping` | `lib/report/confirmMappingApi.ts` | `analysis/router.py` → `validate_mapping` | Step5 |
| POST `/api/validate-data` | `hooks/useDataValidation.ts` | `analysis/validation_router.py` → `validate_dataset` | Step6 |
| POST `/api/evaluate` | `hooks/useReportData.ts` | `evaluation/router.py` → `run_evaluation_pipeline` | 리포트 S1 |
| POST `/api/narrative/jobs`, GET `/api/narrative/jobs/{id}` | `lib/report/narrativeJobApi.ts` (payload=`buildFactSheet.ts`) | `narrative/router.py` → `jobs.create_job`·`read_job` → `generator.generate_sections` | 리포트 S2 |
| POST `/api/generate-narrative` (옛 경로) | 없음 — 운영 프론트는 2026-10-23 배포부터 부르지 않는다(옛 클라이언트용으로 얼려 둠) | `narrative/router.py` → `legacy.generate_narrative` | 리포트 S2(옛) |
| GET `/api/organization`, POST `/api/reports/issue`·`{no}/reissue`, GET `/api/reports/{no}`·`/api/reports/{no}/content` | `lib/report/issuanceApi.ts` · `hooks/useIssuance.ts` | `issuance/router.py` → `service` | 발급 |

- **`/api/validate-data` 와 `/api/evaluate` 는 동일 요청 구조** `EvaluateRequest`(`task_type, column_mappings[], selected_metric_ids[], metadata, beta`), multipart(`file` + `data`=JSON).
- **role 어휘 변환(drift 방지 단일 출처 2함수)**: 요청 조립 `src/lib/mapping/translateRoleToBackend.ts`(task_type 의존), 응답 역변환 `translateRoleToFrontend.ts`. 백엔드 대응물은 `app/core/schemas.py` 의 `ColumnRole` / `VALID_ROLES_BY_TASK` / `METRIC_REQUIREMENTS`(= SPEC.md의 코드 구현).
  - 예: `y_true`→(binary/multiclass) `y_true` · (multilabel) `true_labels`; `score`→`score_positive`(binary); `prob_class_*`→`prob_per_class`(multiclass); `prob_label_*`→`score_per_label`(multilabel); 공통 `id`→`sample_id`, `latency`→`latency`, `ignore`→`ignore`.
- **graceful 처리**: 조회(GET) 실패 시 `null`, 발급(POST) 실패 시 백엔드 `detail` throw. 서술은 무키·호출 실패면 칸마다 규칙 문장으로 **완료**, 작업이 실패(시간 초과·유실)하면 프론트가 "서술 다시 만들기"를 주고 끝나지 않은 서술은 인쇄·발급을 잠근다. snake_case→camelCase 및 KST 표기는 프론트에서 변환.

---

## 5. 문서 색인 (언제 무엇을 읽나)

| 문서 | 언제 읽나 |
|---|---|
| `Capstone_Front/SPEC.md` ⭐ | **지표 요구사항·컬럼 역할·검증 규칙·task_type별 허용 지표를 다루는 모든 작업의 최우선 기준.** 변경 전 프론트/백 양쪽을 이 문서에 맞춘다. |
| `Capstone_Back/docs/ARCHITECTURE.md` | 백엔드 온보딩·구조 기준 — `app/` 도메인 구조, router→service→순수로직 패턴, 새 지표/API/테이블 추가 위치. (`Capstone_Back/README.md`는 스택·주요 API·로컬 실행 요약) |
| `Capstone_Back/docs/ARCHITECTURE.md` 의 §10 · §11 · §12 | **서술(7·8·9절)** 작업 번호·칸별 검문·규칙 템플릿·결정 번호 색인과 왜 그렇게 만들었는지(§10), **발급** 번호 규칙·정정 발급·보관본(§11), **배포·운영** 실제 주소·배포 경로·환경변수·운영 메모(§12). |
| `Capstone_Front/docs/ARCHITECTURE.md` | 프론트 처음 진입 — 페이지=조립자, `src/` 계층, Zustand. 성적서 서술 흐름·사실 시트·판(contentSchema)·고정 문구 번호 표는 §6. |
| `Capstone_Front/docs/FRONTEND_DEVELOPMENT_GUIDELINE.md` | 프론트에 새 페이지/컴포넌트/유틸 추가 시 지킬 규칙. |
| `Capstone_Front/docs/MATERIAL_PICKER.md` | 평가 흐름 만들기 화면(작업 유형 → 블록 고르기 → 입력)의 흐름 규칙·용어. |
| `Capstone_Front/docs/COMPOSER_COMPONENTS.md` | 평가 흐름에 넣는 블록(카드)과 항목(필드) 목록 — 코드 정본 `src/data/reportComposer.ts` 를 읽기 쉽게 옮긴 것. |
| `Capstone_Front/docs/UI_DESIGN.md` | 세 분류 유형을 아이콘으로 나눈 진입 화면 설계. |
| `Capstone_Front/docs/imports/` | 디자인 기준(색·간격·글꼴·가이드라인) 원본. |

> **문서 역할**: 백엔드 문서는 셋뿐이다 — `README.md`(요약·실행), `docs/ARCHITECTURE.md`(구조와 규칙, 서술·발급·배포의 이유까지), 이 문서(프론트+백 지도). 지표·컬럼 규칙은 프론트 저장소의 `SPEC.md` 가 정본이다.

---

## 6. 로컬 실행 / 배포

### 백엔드 (`Capstone_Back`)
```bash
pip install -r requirements-dev.txt        # requirements.txt 포함
cp .env.example .env                        # ANTHROPIC_API_KEY(서술) / OPENAI_API_KEY(매핑) — 없으면 규칙 기반 / DATABASE_URL(없으면 SQLite data/app.db)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # 또는 python -m app.main
venv/bin/python -m pytest                    # 테스트 (asyncio_mode=auto, pythonpath=.) — 맨 pytest 는 다른 파이썬을 잡을 수 있다
```
첫 부팅 시 `init_db()`+`seed_organization()` 자동(발급 표·서술 작업 표). 키 없이도 기동됨.

### 프론트 (`Capstone_Front`)
```bash
pnpm install
pnpm dev            # Vite dev (VITE_API_BASE_URL 미설정 시 상대경로→프록시)
pnpm typecheck && pnpm test && pnpm build
```

### 배포
- **백엔드 = Render** web service. 주소는 `https://capstone-back-59z8.onrender.com`(하위 도메인 `capstone-back-59z8`). 배포는 **CI(deploy job)의 Render Deploy Hook 호출이 유일 경로**다. `render.yaml` 의 값(이름·리전·Python 버전·Start Command 등)은 참고용이고, 실제 설정은 Render 대시보드가 정본이다. `DATABASE_URL`/`OPENAI_API_KEY`/`ANTHROPIC_API_KEY`는 대시보드에만 넣는다(커밋 금지).
- **CI** (`.github/workflows/ci.yml`): PR/push(`dev`,`main`) → test job — `pytest -q` 뒤 서버를 띄워 스모크: `/health` → `/api/organization` → 발급 → 발급정보 조회 → `/content` 원본 조회 왕복 → 서술 작업 `POST /api/narrative/jobs` → `GET` 이 `done` 이 되고 칸 출처가 `fallback`·`fixed` 뿐인지(키 없음) 확인. `main` push 가 통과하면 deploy job 이 포크(`hellinii/Capstone_Back`) `main` 으로 복사 → Deploy Hook 호출 → `/health` 의 `commit` 이 새 커밋이 될 때까지 기다리고 `persistent`(영속 DB)도 확인. 자세히는 `docs/ARCHITECTURE.md` §12.
- **프론트 = Vercel** (`vercel.json`), **DB = Neon PostgreSQL**(`ap-southeast-1`).

---

## 7. 새 세션 추천 읽기 순서
1. 이 문서(PROJECT_OVERVIEW.md)로 전체 지도 파악 →
2. 작업 대상에 따라 §5 색인에서 해당 문서 →
3. 지표/컬럼/검증 관련이면 **반드시 `SPEC.md` 먼저** →
4. 코드 진입: 프론트는 `routes.ts`→`pages/`→`components/`, 백엔드는 `app/main.py`→도메인 `router.py`→`service.py`.
