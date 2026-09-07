# 백엔드 구조 한눈에 보기 (처음 오신 분 먼저 읽기)

이 문서 하나면 **5분 안에** 이 백엔드가 뭘 하고, 코드가 어디 있고, 새 작업을 어디서 시작할지 알 수 있습니다. 코드를 처음 연다는 가정으로 씁니다.

---

## 1. 이 백엔드가 하는 일 (한 문단)

**ISO/IEC TS 4213** 기준으로 AI **분류 모델**의 성능을 평가하고 성적서를 발급하는 FastAPI 서버입니다. 사용자가 모델 예측 결과 파일(CSV/JSON)을 올리면 → ① **어떤 컬럼이 정답/예측/점수인지 파악**하고(LLM 자동 매핑) → ② **지표(정확도·정밀도 등)를 계산**하고 → ③ **결과를 자연어로 설명**하고 → ④ **성적서를 발급**합니다. Render + Neon Postgres에 배포됩니다.

---

## 2. 딱 한 장으로 보는 구조

> **핵심 규칙 하나만 기억하세요: `app/` 아래 폴더 = "하는 일(도메인)" 하나.**
> 그리고 **어느 도메인이든 파일 구성이 똑같습니다** → `router`(API) → `service`(흐름 조율) → 나머지(세부 로직) + `schemas`(그 도메인의 데이터 형태).

```
app/
├── main.py            앱 진입점 (uvicorn app.main:app) — 라우터 5개 연결, 시작 시 DB 준비
│
├── core/              🔧 모든 도메인이 공유하는 것
│   ├── schemas.py       공용 데이터 규격(enum·역할규칙·지표요건·"분석→평가" 인계 모델)
│   ├── database.py      DB 연결(SQLAlchemy) — 로컬 SQLite / 배포 Postgres
│   └── parsing.py       업로드 파일(CSV/JSON) → 표(DataFrame) 변환 (공용 유틸)
│
├── analysis/          ① 분석 — 파일의 어떤 컬럼이 뭔지 파악 + 데이터 점검
│   ├── router.py            API: 컬럼 매핑 (/api/analyze-columns, /api/confirm-mapping)
│   ├── validation_router.py API: 데이터 검증 (/api/validate-data)
│   ├── analysis_service.py    흐름 조율(무키면 규칙매핑, LLM 실패하면 규칙매핑으로 강등)
│   ├── validation_service.py  데이터 검증 흐름 조율
│   ├── llm_mapper.py     LLM으로 컬럼 자동 매핑        fallback_mapper.py  규칙 기반 매핑
│   ├── reconcile.py      LLM이 준 컬럼명을 실제 헤더에 맞춤   metadata.py  클래스·분포 추출
│   ├── prompt_builder.py 매핑용 LLM 프롬프트          validator.py  매핑 유효성·계산가능 지표
│   ├── validation_checks.py  데이터 점검 항목들       schemas.py  이 도메인 데이터 형태
│
├── evaluation/        ② 평가 — 지표 계산
│   ├── router.py        API (/api/evaluate)          service.py  평가 흐름 조율
│   ├── engine.py        지표 → 계산함수로 분배          preprocessor.py  계산 전 데이터 정리
│   ├── report.py        결과 성공/실패 정리           schemas.py
│   └── metrics/         실제 지표 계산(sklearn): common·binary·multiclass·multilabel
│
├── narrative/         ③ 서술 — 결과를 사람이 읽을 글로 (LLM)
│   ├── router.py        API (/api/generate-narrative)  service.py  서술 생성 흐름 조율
│   ├── grounding.py     환각 방어(없는 숫자 못 쓰게)   derived.py  서버 파생 계산
│   ├── prompt.py        서술 프롬프트                 fallback.py  LLM 실패 시 규칙 서술
│   └── baselines.py     지표 참조 기준치              schemas.py
│
└── issuance/          ④ 발급 — 성적서 번호 채번·발급 (유일하게 DB 사용)
    ├── router.py        API (/api/organization, /api/reports/*)  service.py  발급 트랜잭션
    ├── models.py        DB 테이블(Organization·Report·Issuance)  serializers.py  DB→응답 변환
    └── bootstrap.py     기본 기관 시드                schemas.py
```

> 💡 각 폴더의 `__init__.py`를 열면 그 도메인의 **파일 목차**가 주석으로 정리돼 있고, 각 `.py` 파일 맨 위에는 **"이 파일이 뭐고 뭐랑 상호작용하는지"** 헤더 주석이 있습니다.

---

## 3. 요청이 흐르는 순서

사용자가 파일을 올리면 도메인을 **순서대로** 거칩니다. 이 흐름 = 위 폴더 순서입니다.

```
[예측결과 파일 업로드]
      │
      ▼
① /api/analyze-columns   analysis : 컬럼 역할 자동 파악 (LLM, 없으면 규칙)
      ▼
① /api/confirm-mapping   analysis : 사용자가 고른 매핑 검사 → 계산 가능한 지표 목록
      ▼  (선택) /api/validate-data  analysis : 계산 전 데이터 점검(결측·중복 등)
      ▼
② /api/evaluate          evaluation : 지표 계산
      ▼
③ /api/generate-narrative narrative : 계산 결과를 자연어로 서술
      ▼
④ /api/reports/issue     issuance : 성적서 번호 발급 + DB 저장
```

- **LLM(OpenAI)을 쓰는 곳**: ①분석·③서술. 키(`OPENAI_API_KEY`)가 없으면 자동으로 규칙 기반으로 대체돼서 서버는 안 죽습니다.
- **DB를 쓰는 곳**: ④발급 뿐. 나머지는 업로드 파일을 메모리에서 처리하고 끝(디스크 저장 없음).

---

## 4. 도메인 요약표

| 도메인 | 하는 일 | 주요 API | "진입점" 파일 |
|---|---|---|---|
| **core** | 공용 규격·DB·파일파싱 | — | `schemas.py`, `database.py` |
| **analysis** | 컬럼 파악 + 데이터 점검 | `/api/analyze-columns`, `/api/confirm-mapping`, `/api/validate-data` | `router.py` → `analysis_service.py` |
| **evaluation** | 지표 계산 | `/api/evaluate` | `router.py` → `service.py` |
| **narrative** | 결과 서술 | `/api/generate-narrative` | `router.py` → `service.py` |
| **issuance** | 성적서 발급 | `/api/organization`, `/api/reports/*` | `router.py` → `service.py` |

시스템: `GET /health` (상태 확인).

---

## 5. 새 작업을 어디서 시작하나 (신규 개발자용)

| 하고 싶은 것 | 열 파일 |
|---|---|
| API 동작/입출력 바꾸기 | 해당 도메인 `router.py` (얇음 — 검증·상태코드만) |
| 비즈니스 로직(흐름) 바꾸기 | 해당 도메인 `service.py` |
| 요청/응답 필드 추가 | 해당 도메인 `schemas.py` (공용이면 `core/schemas.py`) |
| 새 지표 추가 | `evaluation/metrics/*` + `evaluation/engine.py`의 레지스트리 |
| 서술 문구/규칙 | `narrative/prompt.py`(LLM) 또는 `narrative/fallback.py`(규칙) |
| DB 테이블 | `issuance/models.py` |

**패턴이 항상 같습니다**: 요청은 `router`(HTTP만) → `service`(흐름) → 세부 헬퍼로 흐릅니다. 그래서 어느 도메인이든 같은 방식으로 읽힙니다.

---

## 6. 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload      # → http://127.0.0.1:8000/docs (Swagger로 모든 API 확인)
pytest -q                          # 테스트 (tests/)
```

환경변수: `OPENAI_API_KEY`(없으면 규칙 기반 동작), `DATABASE_URL`(없으면 로컬 SQLite `data/app.db`).

---

## 7. 알아두면 좋은 규칙

1. **import는 절대 경로** (`from app.analysis.llm_mapper import ...`).
2. **의존 방향은 항상 도메인 → core** (core는 도메인을 모름 → 순환 없음). 예외: 평가가 분석의 파싱/검증을 재사용(상류 참조)하는 것은 허용.
3. **`app/__init__.py`·`app/core/__init__.py`에 `import`를 넣지 마세요** — `main.py`의 `load_dotenv()`가 DB import보다 먼저 실행돼야 하는 순서가 깨집니다(파일 주석 참조).
4. **DB 스키마는 `init_db()`의 `create_all`로 생성**(마이그레이션 도구 없음).
5. 배포 구성은 [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) 참조.

---

## 8. 알려진 한계 (2026-09-07 확정)

여기 적힌 것들은 **버그가 아니라 의식적으로 그은 선**이다. 각 항목에 왜 그렇게 정했는지를
남긴다 — 근거를 잃으면 다음 사람이 같은 논의를 처음부터 다시 한다.
대응 항목 번호는 `ISSUES.md` 를 가리킨다.

### 8.1 성적서의 진본성 — 서버가 완전히 보증하지는 않는다 (F-02·F-03·F-05)

발급 시점의 성적서 원본(JSON + SHA-256)과 기관 스냅샷은 서버에 보관된다. 번호로
복원할 수도 있다. 그러나 **다음 셋은 구현하지 않았다.**

| # | 한계 | 왜 이번 범위에서 제외했나 |
|---|---|---|
| **F-02** | 평가와 발급이 서버에서 결속되지 않는다 — 평가를 거치지 않고도 정식 번호를 받을 수 있다 | 백엔드 단독으로 닫히지 않는다(프론트 필수·3단 롤아웃). 무엇보다 **기존 사용자 전원이 발급 불가**가 되고, 결속을 넣어도 **2행짜리 CSV 로 평가를 통과시켜 우회**할 수 있어 실효가 낮다 |
| **F-03** | `/api/generate-narrative` 가 클라이언트가 보낸 `fact_sheet` 를 무검증 신뢰한다 | F-02 와 같은 뿌리다. 서버가 평가 결과를 갖고 있어야 대조할 수 있는데 그것이 F-02 다 — 따로 닫으면 반쪽이 된다 |
| **F-05** | 서버가 PDF 를 만들지도 보관하지도 않는다. 최종 산출물은 브라우저 인쇄물이다 | 서버 렌더링 스택(헤드리스 브라우저)이 필요하고 Render free 512 MB 안에서 운영 위험이 크다 |

**실질적 의미**: 발급된 성적서의 *내용*은 서버 사본으로 검증할 수 있지만, *그 성적서가
실제 평가를 거쳤는가*는 서버가 보증하지 않는다. 대외 제출용으로 쓸 때 이 점을 알고 써야 한다.

### 8.2 운영·인프라

| # | 한계 | 판단 |
|---|---|---|
| **G-03** | LLM 예산 상한이 **프로세스 로컬**이라 워커가 N개면 실효 상한도 N배 | 과허용 방향으로만 틀리므로 정상 사용자를 잘못 막지 않는다. 정확한 총량이 필요해지면 DB 카운터로 옮긴다 |
| **G-03** | 인증·레이트리밋 없음 | 인증은 넣지 않기로 확정(★결정 9). IP 리밋은 **프록시 뒤 실제 client IP 가 미관측**이라 보류 — 모든 요청이 같은 IP 로 보이면 IP 리밋이 전역 리밋으로 붕괴해 정상 사용자 1명이 전체를 막는다 |
| **F-11** | 마이그레이션 도구가 없다(`create_all` 만) | **기존 테이블에 컬럼을 추가하지 않는다**를 규칙으로 둔다. 신규 테이블은 `create_all` 이 정상 생성하므로(실증) 스키마 확장은 새 테이블로 한다 |
| **F-12** | 동시 채번 직렬화(`BEGIN IMMEDIATE`)가 SQLite 전용이라 PostgreSQL 에는 적용되지 않는다 | 재시도 5회 + 유니크 제약이 실질 방어선이다. 동시 발급이 드물어 현행 유지 |
| **H-06** | PostgreSQL 경로가 테스트·CI 어디서도 실행되지 않는다 | 로컬에 `psycopg2` 가 없고 CI 에 DB 서비스를 붙이지 않았다. **로컬 초록은 배포 조합의 증거가 아니다** |
| **H-05** | 로컬 pytest 가 프로젝트 venv 가 아니라 전역 anaconda 에서 돈다(fastapi/pydantic 버전이 `requirements.txt` 와 다름) | 위와 같은 이유로 **배포 조합의 증거는 CI 뿐**이다 |
| **G-07** | `DATABASE_URL` 이 없으면 휘발성 SQLite 로 내려가고 기동을 계속한다 | 하드 실패 가드는 `REQUIRE_PERSISTENT_DB=1` 로 **기본 꺼짐**이다. 프로덕션이 이미 SQLite 로 돌고 있다면 켜는 순간 서비스가 죽기 때문. 대신 `/health` 가 평시에도 `persistent` 를 싣고 기동 시 경고를 남긴다 |

### 8.3 서술(LLM) 품질

| # | 한계 | 판단 |
|---|---|---|
| **G-05** | grounding 이 **숫자 토큰**을 대조한다. 판정과 직접 모순되는 주장은 별도로 잡지만, 완곡한 표현이나 영어 서술은 잡지 못한다 | 일반적인 '금지 표현 사전'은 문체 취향이라 도입하지 않는다. 사실 관계(판정 모순)만 막는다 |

### 8.4 프론트 측 한계 (`Capstone_Front`)

| # | 한계 | 판단 |
|---|---|---|
| **E-10** | 워크스페이스 저장소에 크로스탭 동기화가 없어 두 탭을 함께 쓰면 나중 쓰기가 상대 탭의 run 을 덮는다 | 단일 탭 사용이 정상 흐름이다. `storage` 이벤트 동기화는 다음 범위 |
| **E-11** | run 하나당 수십만 바이트가 localStorage 에 쌓여 quota 초과 시 저장이 실패할 수 있다 | 저장 실패가 진행 중인 입력을 잃게 하지는 않는다(try/catch). 서버 보관본이 생겼으므로(F-01) 위험은 종전보다 낮다 |
| **E-15** | 평가·검증 요청에 타임아웃·재시도가 없다(컬럼 분석에만 있다) | 평가·검증은 사용자가 결과를 기다리는 전면 작업이라 조용한 재시도가 오히려 혼란스럽다. 현행 유지 |
| **E-14** | LLM 서술 생성 중(최대 160초) 이탈했다가 재진입하면 평가와 서술이 다시 실행된다 | 캐시 히트 조건이 `isEvaluated`(서술 병합 완료)라 그렇다. 미완 상태를 캐시하면 반쪽 성적서가 완성본으로 굳는다 — 재실행이 안전한 쪽이다 |
