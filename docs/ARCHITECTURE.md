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
