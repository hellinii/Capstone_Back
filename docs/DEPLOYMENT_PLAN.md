# 배포 전환 계획 — Render(무료) + Neon Postgres + Vercel

> **상태**: 계획 확정, 구현 착수 전
> **작성일**: 2026-07-07
> **대상 독자**: 이 문서만 보고 배포를 진행/이어받을 팀원
> **관련 레포**: 백엔드 [Capstone_Back](https://github.com/hwanginyong02/Capstone_Back) · 프론트 [Capstone_Front](https://github.com/hwanginyong02/Capstone_Front)

---

## 0. 배경 및 결정 사항

### 왜 Render + Neon인가 (2026-07 기준 무료 정책 비교)

| 항목 | **Render** | Railway | Fly.io |
|---|---|---|---|
| 기한 없는 무료 티어 | ✅ 무료 웹서비스 (월 750시간) | ❌ 체험 $5(1회) 후 월 $1 크레딧뿐 | ❌ 신규 계정은 2시간/7일 체험만 |
| 실질 최소 비용 | **$0** | 사실상 $5/월 (Hobby) | ~$2/월 (완전 종량제) |
| 콜드스타트 | 15분 유휴 시 중지 → 복귀 ~30–60초 | 없음 | 없음 |
| 무료 티어 영구 디스크 | ❌ 불가 | 유료 | 유료 |

- 무료가 최우선 조건 → **Render**가 유일한 선택지.
- 단, Render 무료 티어는 **영구 디스크를 못 붙임** → 현재 SQLite(`data/app.db`)로는 **재배포/재시작마다 발급 이력(채번)이 초기화**됨.
- Render 자체 무료 Postgres는 30일 후 만료라 부적합 → **Neon 무료 Postgres**(0.5GB, 만료 없음, 유휴 시 자동 절전 후 접속 시 자동 재개)를 사용.

### 확정된 결정 3가지

| # | 결정 | 선택 | 근거 |
|---|---|---|---|
| 1 | DB | **Neon** (Supabase 아님) | Supabase 무료는 7일 무활동 시 프로젝트 일시정지 + 수동 복구 필요. Neon은 접속만 하면 자동 재개(~1초) |
| 2 | 프론트 연동 | **`VITE_API_BASE_URL` 직접 호출** (Vercel rewrite 프록시 아님) | Vercel 프록시는 응답 ~30초 제한 → Render 콜드스타트(~1분)·느린 LLM 응답에서 504 위험. 직접 호출은 브라우저가 끝까지 대기 |
| 3 | Render 추적 브랜치 | **main** (머지 후 연결) | main 푸시마다 자동 배포되는 표준 구성. ⚠️ 현재 origin/main은 PR #4 시점으로 오래됨 → **Render 연결 전 최신 작업 브랜치를 main에 머지 필수** |

---

## 1. 현재 상태 진단 (탐색 완료)

### 이미 전환 친화적인 것 (수정 불필요)

- **채번 동시성 로직이 Postgres 호환**: `services/issuance.py:103-144` — `UNIQUE(year, seq)`(`models.py:59`) + `UNIQUE(report_id, version)`(`models.py:79-81`) 제약과 `IntegrityError`/`OperationalError` 재시도(최대 5회, 시도 간 `rollback()`)로 방어. SQLite의 `BEGIN IMMEDIATE` 직렬화에만 의존하지 않음.
- **SQLite 전용 코드는 전부 게이트됨**: `database.py`의 PRAGMA/BEGIN IMMEDIATE 리스너·`makedirs`는 `_IS_SQLITE`일 때만 실행 → Postgres URL이면 자동으로 건너뜀.
- **런타임 파일 쓰기 없음**: 업로드 CSV는 `await file.read()` → `io.BytesIO` 인메모리 처리. `Data/`는 테스트 전용 픽스처(커밋됨). → Render의 휘발성 디스크에서 문제없음.
- **모델 타입 전부 이식 가능**: Integer/String(길이 없음)/DateTime(naive UTC)/FK만 사용. JSON 컬럼·server default·rowid 의존 없음.
- **스타트업 훅**: `main.py` lifespan에서 `init_db()`(create_all) + `seed_organization()` → Neon 첫 부팅 시 테이블 자동 생성 + 기관 시드. 수동 마이그레이션 불필요.
- **CORS**: `allow_origins=["*"]`, credentials 미사용 → Vercel 프로덕션/프리뷰 URL 모두 즉시 동작. 변경 불필요.
- `GET /health` 엔드포인트 존재 → Render 헬스체크로 사용.

### 바꿔야 할 것 (이 문서의 나머지 전부)

| 문제 | 위치 | 해결 |
|---|---|---|
| Postgres 드라이버 없음 | `requirements.txt` | `psycopg2-binary` 추가 (§3.3) |
| Neon 절전 시 stale 커넥션 | `database.py` 엔진 | `pool_pre_ping` + `pool_recycle` (§3.1) |
| `.env`의 `DATABASE_URL`이 무시되는 버그 | `main.py:14-17` | `load_dotenv()` 순서 수정 (§3.2) |
| 배포 설정 파일 전무 | 레포 루트 | `render.yaml` 신규 (§3.4) |
| `python main.py`는 포트 8000 하드코딩 | `main.py:73-75` | 시작 커맨드로 uvicorn 직접 실행 (§3.4) — 코드 수정은 안 함 |
| 프론트 API가 전부 상대경로 `/api/...` | 10곳 산재 | `apiUrl()` 헬퍼 + 1줄 래핑 (§4) |

### 로컬 개발은 아무것도 안 바뀜

- 백엔드: `DATABASE_URL` 미설정 → 지금처럼 SQLite(`data/app.db`)
- 프론트: `VITE_API_BASE_URL` 미설정 → `""` → 상대경로 → Vite dev 프록시(`127.0.0.1:8000`) 그대로
- 기존 테스트(SQLite 픽스처, `configure_sqlite` 직접 호출)도 그대로 green이어야 함

---

## 2. 목표 아키텍처

```
[프로덕션]
  브라우저 ── Vercel (정적 SPA, VITE_API_BASE_URL 빌드타임 주입)
     │  fetch("https://capstone-back.onrender.com/api/...")  ← 크로스오리진 직접 호출
     ▼
  Render 무료 웹서비스 (uvicorn, main 브랜치 자동 배포, /health 헬스체크)
     │  DATABASE_URL (postgresql://...sslmode=require)
     ▼
  Neon 무료 Postgres (ap-southeast-1, direct 연결, 유휴 시 자동 절전/재개)

[로컬 개발 — 기존과 동일]
  vite dev (/api 프록시) → uvicorn :8000 → SQLite data/app.db
```

환경 전환은 env var 2개가 전부: `DATABASE_URL`(백엔드, 런타임), `VITE_API_BASE_URL`(프론트, **빌드 타임**).

---

## 3. 백엔드 코드 변경 (Capstone_Back)

### 3.1 `database.py` — 엔진 블록 교체 (13–22행 부근)

변경 이유:
- **스킴 정규화**: SQLAlchemy 2.0은 legacy `postgres://` 스킴을 거부. Neon은 `postgresql://`을 주지만 Render/Heroku류 URL 대비 방어적으로 처리.
- **Neon 절전 대응**: Neon free는 ~5분 유휴 시 compute를 절전 → SQLAlchemy 풀에 남은 커넥션이 죽음. `pool_pre_ping=True`(checkout 시 검증·투명 재연결) + `pool_recycle=300`(절전 윈도우보다 오래된 커넥션 선제 폐기).

```python
# ── 엔진 ──────────────────────────────────────────────────────────────────────
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

# Heroku/Render 스타일 URL 은 legacy "postgres://" 스킴을 쓰는데 SQLAlchemy 2.0 은
# 이를 거부한다. (Neon 은 postgresql:// 을 주지만 방어적으로 정규화.)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

_IS_SQLITE = DATABASE_URL.startswith("sqlite")

if _IS_SQLITE:
    _ENGINE_KWARGS: dict = {
        # check_same_thread=False: FastAPI 스레드풀에서 세션 사용 허용(요청별 세션이라 안전).
        "connect_args": {"check_same_thread": False},
    }
else:
    _ENGINE_KWARGS = {
        # Neon free tier 는 ~5분 유휴 시 compute suspend → 풀에 남은 연결이 죽는다.
        "pool_pre_ping": True,   # checkout 시 연결 검증, 죽었으면 투명하게 재연결
        "pool_recycle": 300,     # suspend 윈도우(5분)보다 오래된 연결은 선제 폐기
    }

engine = create_engine(DATABASE_URL, **_ENGINE_KWARGS)
```

`configure_sqlite` / `if _IS_SQLITE: configure_sqlite(engine)` / `SessionLocal` / `Base` / `get_db` / `init_db` / `seed_organization`은 **그대로 둔다** (테스트가 `configure_sqlite`를 직접 import·호출함).

### 3.2 `main.py` — dotenv 로드 순서 버그 수정

**확인된 버그**: `main.py:14`의 `from database import ...`가 `main.py:17`의 `load_dotenv()`보다 먼저 실행되는데, `database.py`는 **import 시점**(모듈 바디)에 `DATABASE_URL`을 읽는다 → `.env`에 넣은 `DATABASE_URL`이 현재 조용히 무시됨. (`OPENAI_API_KEY`는 lifespan 안에서 늦게 읽어서 무사했음.)

```python
from dotenv import load_dotenv

load_dotenv()  # database.py 가 import 시점에 DATABASE_URL 을 읽으므로 그보다 먼저 실행해야 함

from database import init_db, seed_organization
from routers import analyze, evaluate, validate, narrative, reports
```

(기존 17행의 `load_dotenv()`는 삭제.)

추가(권장): lifespan의 `init_db()` 다음에 DB 백엔드 로그 1줄 — Render에서 `DATABASE_URL` 누락 시 로그에서 즉시 발견하기 위함.

```python
from database import DATABASE_URL, init_db, seed_organization
...
    init_db()
    seed_organization()
    print(f"✅ 발급 메타 DB 초기화 완료 (backend={'sqlite' if DATABASE_URL.startswith('sqlite') else 'postgresql'})")
```

**변경하지 않는 것**: CORS(`["*"]` 유지 — credentials 미사용이라 안전, Vercel 프리뷰 URL까지 커버), `__main__` 블록(프로덕션은 startCommand가 uvicorn을 직접 실행하므로 미사용).

### 3.3 `requirements.txt` — 1줄 추가

```
psycopg2-binary==2.9.10
```

- SQLAlchemy의 `postgresql://` 기본 방언이 psycopg2 → **URL에 `+드라이버` 접미사 불필요**, Neon 문자열 그대로 사용.
- cp312 manylinux 휠 제공 → Render 빌드에서 컴파일 없음.
- 번들 libpq가 Neon의 `channel_binding=require` 파라미터 지원.

### 3.4 `render.yaml` — 레포 루트에 신규 (Blueprint 배포)

대시보드 수작업 대신 Blueprint를 쓰는 이유: 서비스 정의가 PR로 리뷰되고, 재현 가능하며, 남는 수작업은 시크릿 2개 입력뿐.

```yaml
services:
  - type: web
    name: capstone-back
    runtime: python
    plan: free
    region: singapore          # 사용자·Neon(ap-southeast-1)과 가장 가까움
    branch: main
    healthCheckPath: /health
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.12.13         # 로컬(.venv)과 동일 버전 핀
      - key: DATABASE_URL
        sync: false            # Neon 연결 문자열 — 대시보드에서 입력(커밋 금지)
      - key: OPENAI_API_KEY
        sync: false            # 대시보드에서 입력(커밋 금지)
```

⚠️ 시작 커맨드로 `python main.py`를 쓰면 안 됨 — 포트 8000 하드코딩 + `reload=True`라 Render의 `$PORT` 헬스체크가 실패한다.

### 3.5 `.env.example` — DATABASE_URL 문서화 추가

```
# ── 데이터베이스 (선택) ──
# 미설정 시 로컬 SQLite(data/app.db) 사용 — 로컬 개발 기본값.
# 배포(Neon Postgres)용 또는 로컬에서 Neon 을 테스트할 때만 설정.
# Neon 대시보드의 "직접(direct)" 연결 문자열을 그대로 붙여넣기 (sslmode=require 유지):
# DATABASE_URL=postgresql://<user>:<password>@ep-xxxx.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
```

### 변경하지 않는 파일

`models.py`, `services/issuance.py`, `routers/*`, 테스트 전체 — Postgres 호환 확인 완료(§1).

---

## 4. 프론트 코드 변경 (Capstone_Front)

### 4.1 `src/lib/apiBase.ts` — 신규

```ts
/**
 * API base URL.
 * - 프로덕션(Vercel): VITE_API_BASE_URL = "https://capstone-back.onrender.com"
 *   (빌드 시점에 주입 — Vercel 프로젝트 환경변수, 값 변경 시 재배포 필요)
 * - 로컬 dev/테스트: 미설정 → "" → 상대경로 "/api/..." 가 Vite 프록시로 전달(기존과 동일)
 */
const raw = import.meta.env.VITE_API_BASE_URL ?? "";
export const API_BASE = raw.replace(/\/+$/, ""); // 끝 슬래시 제거(중복 // 방지)

/** "/api/..." 경로 앞에 백엔드 origin 을 붙인다. */
export function apiUrl(path: string): string {
  return API_BASE + path;
}
```

### 4.2 `src/vite-env.d.ts` — env 타입 추가

```ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 백엔드 origin (예: "https://capstone-back.onrender.com"). 미설정 시 dev 프록시 사용. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
```

### 4.3 fetch 호출부 10곳 래핑 (7개 파일)

각 파일에 `import { apiUrl } from "@/lib/apiBase";` 1줄 추가(`@` 별칭은 `vite.config.ts:14-17`에 설정돼 있음), 호출부는 1줄 수정:

패턴: `fetch("/api/x", …)` → `fetch(apiUrl("/api/x"), …)`

| 파일:행 | 엔드포인트 | 비고 |
|---|---|---|
| `src/pages/DataUpload.tsx:39` | POST `/api/analyze-columns` | FormData 업로드 |
| `src/pages/ColumnMapping.tsx:53` | POST `/api/confirm-mapping` | JSON |
| `src/pages/DataValidation.tsx:93` | POST `/api/validate-data` | FormData 업로드 |
| `src/hooks/useReportData.ts:116` | POST `/api/evaluate` | FormData 업로드 |
| `src/lib/report/fetchNarrative.ts:48` | POST `/api/generate-narrative` | LLM, 느림 |
| `src/lib/report/issuanceApi.ts:118` | GET `/api/organization` | |
| `src/lib/report/issuanceApi.ts:129` | GET `/api/reports/{reportNo}` | 템플릿 리터럴 |
| `src/lib/report/issuanceApi.ts:139` | POST `/api/reports/issue` | |
| `src/lib/report/issuanceApi.ts:160` | POST `/api/reports/{reportNo}/reissue` | 템플릿 리터럴 |
| `src/hooks/usePdfDownload.ts:11` | GET `/api/reports/{id}/pdf` | blob 다운로드 — 동작 동일 |

FormData·JSON·blob 전부 URL 문자열만 바뀌므로 동작 변화 없음. env 미설정 시 `apiUrl`은 입력을 그대로 반환 → vitest·dev 프록시가 보는 URL이 기존과 바이트 단위로 동일.

`vite.config.ts` / `vercel.json`은 **변경 없음**.

---

## 5. 진행 순서 (운영 런북)

범례: 👤 = 대시보드/브라우저 수작업, 🤖 = 코드·CLI 작업

| # | 담당 | 작업 |
|---|---|---|
| 1 | 🤖 | Capstone_Back에 §3 적용 → 커밋 (작업 브랜치) |
| 2 | 🤖 | Capstone_Front에 §4 적용 → 커밋 (작업 브랜치) |
| 3 | 🤖 | **로컬 검증(SQLite 경로)**: 백엔드 `pytest` 전부 green / 프론트 `pnpm typecheck && pnpm test && pnpm build` green / `pnpm dev`로 전체 플로우 기존과 동일 확인 |
| 4 | 👤 | **Neon 생성**: [neon.tech](https://neon.tech) GitHub 로그인 → New Project → 리전 **AWS ap-southeast-1 (Singapore)**, DB 이름 기본값(`neondb`) → Connect 패널에서 **Direct connection** 선택(❗`-pooler`가 붙은 Pooled 아님) → 연결 문자열 복사 (`postgresql://...?sslmode=require&channel_binding=require`) |
| 5 | 🤖 | **로컬 검증(Postgres 경로)**: `.env`에 `DATABASE_URL=<Neon 문자열>` 추가(§3.2 수정 덕에 동작) → `uvicorn main:app --port 8000` → §6 체크 → 끝나면 `.env`에서 제거(로컬 SQLite 복귀) |
| 6 | 🤖👤 | 두 레포 push → PR 생성 → **main에 머지**. ⚠️ origin/main이 PR #4 시점으로 오래됐으므로 최신 작업 브랜치 전체가 main에 들어가야 함. render.yaml이 main에 있어야 다음 단계 가능 |
| 7 | 👤 | **Render 생성**: [render.com](https://render.com) GitHub 로그인 → New + → **Blueprint** → `Capstone_Back` 연결 → render.yaml 자동 인식 → 프롬프트에 `DATABASE_URL`(4단계 문자열)·`OPENAI_API_KEY` 입력 → Apply → 빌드 2~5분 |
| 8 | 👤🤖 | **배포 확인**: `https://capstone-back.onrender.com/health` → `{"status":"ok"}` / `/docs` Swagger 로드 / Render 로그에 `backend=postgresql` 라인 / Neon 콘솔 Tables에 `organization`·`report`·`issuance` + 기관 시드 1행 |
| 9 | 👤 | **Vercel 연결**: Capstone_Front 프로젝트 → Settings → Environment Variables → `VITE_API_BASE_URL` = `https://capstone-back.onrender.com` (끝 슬래시 없음, 스코프 Production — 프리뷰도 실서버를 치게 하려면 Preview 포함) → **Redeploy 필수** (빌드 타임 주입) |
| 10 | 👤🤖 | **E2E 스모크** (§6 배포 후 체크리스트) |
| 11 | 👤 | (선택) **콜드스타트 방지**: [uptimerobot.com](https://uptimerobot.com) 무료 → HTTP 모니터 `https://capstone-back.onrender.com/health`, 5분 간격. Render 무료 750시간/월 ≥ 24/7(≈744h)이라 상시 워밍 가능 |

---

## 6. 검증 체크리스트

### 머지 전 — SQLite 경로 (회귀 없음 확인)
- [ ] `pytest` 전부 green — 변경분은 비-SQLite 분기뿐, 기존 락킹 테스트(`test_issuance.py`) 무영향
- [ ] `pnpm typecheck` / `pnpm test` / `pnpm build` green
- [ ] `pnpm dev` + 로컬 백엔드로 업로드→평가→발급 플로우 기존과 동일

### 머지 전 — 로컬 앱 + 실제 Neon
- [ ] `/health` ok, 기동 로그 `backend=postgresql`, Neon 콘솔에 테이블 3개 + 기관 시드 1행 (create_all + seed 동작)
- [ ] 앱 재기동 → 에러 없이 부팅 (create_all 멱등, 시드 스킵)
- [ ] 발급 2회(`run_id` 다르게) → `seq` 증가 / 같은 `run_id` 재호출 → 동일 발급본(멱등) / reissue → version 증가
- [ ] Neon 콘솔에서 `uq_report_year_seq`, `uq_issuance_report_version` 제약 존재 확인 (동시 채번 경합을 재시도 가능한 IntegrityError로 바꿔주는 방어선)
- [ ] **절전 드릴**: 5분+ 방치(Neon 대시보드에서 compute suspend 확인) → `GET /api/organization` → 500 없이 성공 (pre_ping 재연결, 1~3초 지연은 정상)

### (선택) Vercel 전 크로스오리진 리허설
- [ ] 로컬에서 `VITE_API_BASE_URL=<Render URL>`로 `pnpm build && pnpm preview` → localhost:4173에서 플로우 동작 → CORS·base URL을 Vercel 없이 검증

### 배포 후 — Vercel URL에서
- [ ] 첫 요청 전 `/health` 워밍 (콜드스타트 ~30–60초 감안)
- [ ] CSV 업로드 → 컬럼 매핑 → 검증 → 평가 → 리포트 렌더
- [ ] LLM 서술 생성 (최대 ~45초 — 클라이언트 타임아웃 160초로 콜드스타트+LLM 커버)
- [ ] 발급 → 번호/차수 부여 → 번호로 재조회 → 재발급 시 version 증가
- [ ] PDF 다운로드 정상
- [ ] DevTools Network: 모든 `/api` 요청이 `capstone-back.onrender.com`으로, 콘솔에 CORS 에러 없음

---

## 7. 리스크 및 대응

| 리스크 | 내용 | 대응 |
|---|---|---|
| Render 콜드스타트 | 15분 유휴 → 중지, 웨이크 ~30–60초. 타임아웃 없는 fetch는 그냥 오래 걸리는 것처럼 보임 | §5-11 UptimeRobot 키핑 + 데모 직전 `/health` 워밍. UI 스피너/타임아웃 추가는 이번 범위 아님 |
| `DATABASE_URL` 누락 | 누락 시 휘발성 디스크의 SQLite로 **조용히 폴백** → 재배포마다 데이터 소실 | 3중 방어: `sync: false`(대시보드 입력 강제) + §3.2 기동 로그 + §5-8 Neon 테이블 확인 |
| Neon `channel_binding=require` | psycopg2-binary 번들 libpq가 지원하므로 유지 | 만약 connect 에러에 "invalid connection option channel_binding"이 보이면 그 파라미터만 제거 (`sslmode=require`로 충분) |
| 마이그레이션 도구 없음 (의도적) | `create_all`은 새 테이블만 생성 — 기존 테이블 컬럼 변경은 반영 안 됨 | 추후 스키마 변경 시 Neon SQL 에디터에서 수동 `ALTER`. 캡스톤 규모에 적정 (설계 문서 §11.1과 일치) |
| Vite env는 빌드 타임 고정 | Render URL이 바뀌면 Vercel env 수정만으로는 반영 안 됨 | env 수정 + **Redeploy** 둘 다 수행 |
| Neon 무료 한도 | 0.5GB 저장/자동 절전 | 데모 규모 대비 충분. direct 연결 한도(~112) ≫ QueuePool 최대 15 |

---

## 부록 — 참고 링크

- Render 무료 티어: https://render.com/docs/free · Blueprint(render.yaml): https://render.com/docs/blueprint-spec
- Neon 무료 플랜/연결: https://neon.tech/docs/connect/connect-from-any-app
- SQLAlchemy 커넥션 풀(pre_ping): https://docs.sqlalchemy.org/en/20/core/pooling.html
- Vite env 변수(빌드 타임): https://vitejs.dev/guide/env-and-mode
- 사내 설계 문서: `docs/ISSUANCE_DB_DESIGN.md` §2(DB 선택)·§4(동시성)·§8(초기화)·§11(마이그레이션)
