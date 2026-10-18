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

---

## 9. 입력 규칙과 상한 — 프론트와 맞추는 값 (2026-10-19, 평가 흐름 만들기 라운드)

프론트(`Capstone_Front`)가 같은 값을 쓰는 규칙은 아래 자리에 **한 번만** 둔다. 프론트 테스트가 이 파일들을
읽어 대조하므로(옆 저장소가 없으면 건너뜀) 값을 바꾸면 프론트도 함께 바꾼다.

| 규칙 | 값 | 자리(백엔드) |
|---|---|---|
| 서술 요청 고객 입력 상한 | 20,000자. **공백 없는 JSON**(`","`·`":"`, 한글 그대로)으로 잰다 — 프론트 `JSON.stringify` 와 같은 수. 프론트는 상한의 90%(18,000자)로 맞춰 보낸다 | `app/narrative/schemas.py` `MAX_CLIENT_INPUTS_CHARS`·`client_inputs_length` |
| 발급 비고·발급자·정정 사유 상한 | 각각 200자(앞뒤 공백을 뗀 길이). 넘으면 422, `detail[].msg` 가 한국어("발급 비고는 200자 이하로 적어 주세요(지금 201자).", "정정 사유를 적어 주세요."). 공백뿐인 비고는 '최초 발급', 공백 발급자는 기관 기본값 | `app/issuance/schemas.py` `MAX_ISSUANCE_TEXT_CHARS` |
| 업로드 크기 상한 | 20 MiB(20 × 1024 × 1024 바이트). 413 문구 "파일이 너무 큽니다. 평가 파일은 20 MiB 이하만 올릴 수 있습니다." | `app/core/upload.py` `MAX_UPLOAD_BYTES`·`format_mib` |
| 확률 컬럼 이름 접두어 | `prob_`·`probability_`·`score_`·`p_` + 클래스·레이블 이름(예: `prob_cat`, `score_sports`) | `app/evaluation/preprocessor.py` `PROB_COLUMN_PREFIXES` |
| 작업 유형 한국어 이름 | 이진 분류 / 다중 클래스 분류 / 다중 레이블 분류. AI 서술 지시문에 이 이름이 간다. 이미 발급된 성적서는 저장된 이름 그대로다 | `app/narrative/prompt.py` `TASK_TYPE_LABELS` |
| 새 검증 항목 이름 | `Probability column names`(error) · `Latency unit`(warning). 프론트 검증 화면이 한국어 이름 표로 바꿔 보인다 | `app/analysis/validation_checks.py` `PROBABILITY_COLUMN_NAMES_CHECK`·`LATENCY_UNIT_CHECK` |

### 9.1 규칙 폴백 매퍼 (OpenAI 키가 없거나 LLM 이 실패할 때)

`app/analysis/fallback_mapper.py` 의 `fallback_role` 이 컬럼 **이름만** 보고 역할을 고른다. 이름은 낱말로 쪼갠다 —
구분자(`_` `-` 공백 `.`)와 camelCase 경계(`imageId` → image, id). 순서대로 보고 처음 맞는 것을 쓴다.

1. 정답 이름(`y_true`·`label`·`target`·`ground_truth`·`actual`·`true_label(s)` …) → `y_true`(멀티레이블은 `true_labels`)
2. 예측 이름(`y_pred`·`pred`·`prediction`·`predicted`·`pred_label(s)` …) → `y_pred`(멀티레이블은 `pred_labels`)
3. 지연시간 이름(`latency`·`latency_ms`·`inference_time(_ms)`·`response_time(_ms)`·`elapsed_ms`, 또는 낱말 `latency`) → `latency`
4. 낱말 `id`·`idx`·`index`·`uid`·`uuid`, 또는 붙여 쓴 식별자 이름(`sampleid`·`rowid`·`userid`·`fileid` 등 목록) → `sample_id`. `valid_flag`·`width` 처럼 'id' 가 **들어 있기만** 한 이름은 잡지 않는다
5. 낱말 `prob`·`proba`·`probability`·`score`(복수형·숫자 꼬리 `probs`·`scores`·`prob1` 포함), 또는 `p_` 로 시작 → 유형의 확률 역할(`score_positive`·`prob_per_class`·`score_per_label`). 이진은 낱말 `positive` 도. `position` 같은 이름은 잡지 않는다
6. 그 밖은 `ignore` + "규칙으로 판단하지 못했다" 안내

한 컬럼만 받는 역할(샘플 ID·정답·예측·이진 확률·지연시간)을 두 번째 컬럼이 또 잡으면, 첫 컬럼만 그 역할로 두고
나머지는 `ignore` 로 내린 뒤 `column_notes` 에 `status="duplicate_role"` 안내를 싣는다 — 역할이 겹치면 매핑 확정이
막히기 때문이다. 단 이진에서 양성 확률 후보가 여럿(`prob_0`·`prob_1`)이면 기본 양성 클래스(§9.2)
이름이 붙은 컬럼을 남긴다 — 첫 컬럼을 남기면 음성 확률이 양성 확률로 들어가 AUROC 가 조용히 뒤집힌다. 이름이 맞는
후보가 없으면 첫 컬럼. 확률 역할은 클래스·레이블마다 한 컬럼이라 여러 컬럼을 그대로 받는다.

자동 매핑 AI 지시문(`app/analysis/prompt_builder.py`)도 다중·멀티 확률 컬럼을 확률 역할로 안내한다(종전에는
"확률 컬럼도 ignore"라고 적어 y_pred 없는 파일의 예측이 사라졌다). 지시문의 역할 집합은 `VALID_ROLES_BY_TASK` 와 같다.

### 9.2 양성 클래스 기본값

사용자가 양성 클래스를 고르지 않았을 때의 기본값은 `app/evaluation/labels.py` `default_positive_class` 하나에서 정한다.
자동 판단(`analysis/metadata.py`)·지표 계산·확률에서 예측 파생(`preprocessor.py`)이 모두 이 함수를 쓴다. 지표 계산은
`metrics/binary.py` 의 `_resolve_positive_value` 한 곳에서 양성을 정하고, Precision·Recall·F1(`metrics/common.py`
`_get_binary_kwargs`)·신뢰구간·혼동 수 집계도 그 함수를 부른다. 양성이 지정되면 그 표기의 원래 값(§9.4)을, 지정되지
않으면(metadata 없이 API 를 직접 부른 경우) 이 기본값을 쓴다.

1. 숫자면 큰 값(1 > 0)
2. 알려진 짝(yes/no·true/false·positive/negative·spam/ham·malignant/benign·fraud/normal …)이면 정해진 값
3. 그 밖은 정렬 순서(숫자면 수치 순, 아니면 사전 순)의 **뒤쪽** 값 — 임시로 고른 것이라 `positive_class_ambiguous=true` 로 알린다

종전에는 3번에서 자동 판단만 **앞쪽**을 골라 화면에 보인 양성과 성적서 수치의 양성이 달랐다('cat'·'dog' 이면 화면은
cat, 지표는 dog). 이제 셋 다 dog 다. 지난 평가는 저장된 양성 값을 쓰므로 영향이 없다. 양성 미지정일 때
Precision·Recall·F1 만 정렬 뒤쪽 값을 쓰던 예외도 없앴다(fraud/normal 이면 종전엔 그 셋만 normal 을 양성으로 봤다).
프론트는 양성을 늘 보내므로 화면 결과는 바뀌지 않는다.

### 9.3 검증 단계에서 미리 보는 것 (`/api/validate-data`)

- **확률 컬럼 이름**(`Probability column names`, error): 다중·멀티에서 예측 컬럼 없이 확률만 매핑하고, 고른 지표가 예측을
  쓰면 평가는 확률 컬럼 이름으로 클래스를 확정해 예측을 만든다. 이름으로 확정할 수 없으면(예: 클래스가 cat/dog 인데
  컬럼이 prob_a·prob_b) 종전에는 검증을 통과한 뒤 평가에서 400 으로 막혔다. 이제 평가와 **같은 함수**
  (`preprocessor.derivation_source`·`probability_names_problem`)로 검증이 미리 보고 막는다. 예측 컬럼이 있거나 이름이
  맞으면 항목을 내지 않는다. 이진은 확률 컬럼이 하나라 이름을 보지 않는다.
- **지연시간 단위**(`Latency unit`, warning): 지연시간 역할은 ms 로 읽는다. 값이 모두 1 미만이면 초로 적은 것으로 보여
  경고한다. 1 ms 미만으로 도는 모델도 있어 막지 않는다 — 단위는 데이터만으로 확정할 수 없다.

### 9.4 클래스 표기

데이터의 클래스 값과 화면·API 가 주고받는 클래스 이름은 다를 수 있다. 빈 칸이 하나라도 있으면 pandas 가 정수 정답
컬럼을 float 로 읽어 1 이 1.0 이 되고, 문자열 클래스에는 앞뒤 공백(`' cat'`)이 붙어 있을 수 있다.

- **주고받는 이름은 표기 하나**: `app/evaluation/labels.py` `normalize_label` — 정수 값 float 는 소수점 없이(1.0 → `'1'`),
  앞뒤 공백은 뗀다(`' cat'` → `'cat'`). metadata 의 클래스 목록·`positive_class`, 응답의 `derived_prediction.positive_class`·
  `negative_class`, `confusion_counts.positive_class` 가 모두 이 표기다. 그래서 정답이 float 이면 양성은 `'1.0'` 이 아니라
  `'1'` 로 나간다(정수·문자열 정답은 종전과 같다). 이미 저장된 옛 run 의 metadata 에는 `'1.0'` 이 남아 있을 수 있다.
- **원래 값은 `find_label` 로 되찾는다**: `y_true == 양성` 처럼 데이터 값과 비교하는 곳은 `str()` 로 맞춰 보지 않고
  `labels.find_label(값들, 표기)` 로 원래 값(1.0·`' cat'`)을 찾아 쓴다. 쓰는 곳: 이진 예측 파생(`preprocessor.py`),
  `binary._resolve_positive_value`(§9.2). 다중 예측 파생은 같은 규칙으로 만든 표기 → 원래 값 표를 쓴다. 표기를 만들어
  내보내는 쪽(metadata 의 이진 클래스 판단·알려진 정답 목록·확률 컬럼 클래스 확정)은 `normalize_label` 을 쓴다.
- **검증의 클래스 불일치 검사는 숫자 표기만 맞춘다**: `validation_checks.check_class_mismatch` 는 `labels.label_text`
  (정수 값 float → 정수 표기, 공백은 그대로)로 비교한다. 정답 1.0 과 예측 1 은 같은 클래스로 보고, 정답 `' cat'` 과
  예측 `'cat'` 처럼 공백이 다른 것은 계속 경고한다 — 평가는 둘을 다른 클래스로 세기 때문이다. `normalize_label` 은
  `label_text` 에 공백 제거를 더한 것이라 숫자 표기 규칙은 한 곳에 있다.
