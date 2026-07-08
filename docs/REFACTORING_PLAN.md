# 백엔드 디렉토리 리팩토링 실행 가이드 — flat 루트 → 도메인 패키지

> 작성일: 2026-07-08 · 기준 커밋: `origin/main` d472edd
> 목표: 루트에 흩어진 모듈을 기능(도메인) 기준 `app/` 패키지로 재구성하되, **배포가 한 번도 깨지지 않고**, **git 히스토리가 보존되고**, **API 경로(/api/\*)가 완전히 불변**(프론트 Vercel 의존)하도록 이행한다.

---

## 0. 이 문서를 따라하기 위한 전제

| 필요한 것 | 확인 방법 |
|---|---|
| 로컬 Python 3.12 + `pip install -r requirements-dev.txt` | `pytest -q` 가 62개 전부 통과 |
| 팀 레포(hwanginyong02/Capstone_Back) push 권한 | PR 생성 가능 여부 |
| Render 대시보드 접근 (hellinii 계정) | §4의 T0 단계에 필요 |
| 최신 main 기준 작업 | 아래 §3 사전 준비 필수 — **로컬에 오래된 브랜치가 체크아웃돼 있을 수 있음** |

**진행 개요 (PR 3개 + 후속 1개):**

| 순서 | 내용 | 배포 발생 | 리스크 |
|---|---|---|---|
| PR0 | CI 안전망(부팅 검증 + API 경로 계약 테스트) + Render 설정 출처 확인(T0) | O (베이스라인) | 없음 (코드 이동 0건) |
| PR1 | 전체 파일 이동 + import 재작성 + 루트 shim | O | 낮음 (배포 설정 접점 0) |
| PR2 | startCommand 전환 → 검증 → shim 제거 | O ×2 | 낮음 (실패해도 shim이 방어) |
| PR3 (후속) | schemas.py 도메인별 분리 | O | 없음 (순수 Python, CI가 전부 검증) |

각 PR이 main에 머지될 때마다 배포가 일어난다. **직전 배포가 healthy임을 확인(§8)한 뒤에만 다음 PR을 진행한다.**

---

## 1. 배경 — 왜, 그리고 무엇이 위험한가

### 1.1 현재 구조의 문제

루트에 `analyzer.py`, `validator.py`, `narrator.py`, `database.py`, `schemas.py` 등 12개 모듈 + 테스트 6개 + 설정 파일이 평면적으로 섞여 있다. `routers/`, `evaluator/`, `services/` 패키지가 일부 존재하지만 도메인 경계가 디렉토리에 드러나지 않는다.

### 1.2 배포 파이프라인 (건드리면 안 되는 것)

```
팀 레포(hwanginyong02) main 푸시
  → GitHub Actions CI (ubuntu-latest, pytest -q)
  → 통과 시: 포크(hellinii) main으로 force-push 미러
  → Render Deploy Hook 호출 (render.yaml autoDeploy:false — 이 훅이 유일한 배포 경로)
  → Render가 헬스체크(/health) 통과 시 신규 버전 활성화
```

- 헬스체크 실패 시 **Render가 이전 버전을 유지**한다(무정지 실패). 부팅이 아예 안 되는 사고는 서비스 중단으로 이어지지 않는다.
- CI가 **Linux(대소문자 구분 파일시스템)**에서 돈다 — macOS 로컬에서 안 잡히는 대소문자 문제는 CI가 유일한 게이트다.
- 포크 main은 순수 배포 미러다. **직접 커밋 금지** (다음 미러 force-push로 덮인다).

### 1.3 배포를 깨뜨릴 수 있는 3대 지점

1. **`render.yaml`의 `startCommand: uvicorn main:app`** — `main.py`를 옮기면 이 문자열이 깨진다. → 루트 shim으로 방어(§5).
2. **전면 flat import** — 모든 내부 import가 `from analyzer import X` 형태로, 레포 루트가 `sys.path`에 있을 때만 동작한다. 하나라도 빠뜨리면 부팅 시 `ModuleNotFoundError`. → 치환표(§5.3) + grep 검증(§5.6)으로 방어.
3. **Render 서비스가 Blueprint(render.yaml) 관리인지, 대시보드 수동 생성인지 불확실** — 수동 생성이면 repo의 render.yaml을 고쳐도 반영되지 않는다. → T0(§4.3)에서 확정 후 PR2 절차 분기.

### 1.4 코드에 숨어 있는 불변식 (반드시 유지)

- **`load_dotenv()` 호출 순서**: `main.py`에서 `load_dotenv()`가 `from database import ...`보다 **먼저** 실행돼야 한다. `database.py`가 import 시점(모듈 최상단)에 `DATABASE_URL`을 읽기 때문이다. 이게 깨지면 프로덕션(실제 env)은 멀쩡하고 **로컬만 조용히** SQLite로 폴백한다.
- **`database.py` ↔ `models.py` 순환 회피 패턴**: `models.py`는 최상단에서 `from database import Base`, `database.py`는 `init_db()`/`seed_organization()` **함수 내부에서만** models를 지연 import한다. 이 지연 import를 최상단으로 끌어올리면 순환 import로 깨진다.
- **`data/` vs `Data/` 대소문자**: `database.py`의 기본 SQLite 경로는 소문자 `data/app.db`인데 실제 디렉토리는 대문자 `Data/`다. macOS(대소문자 무시)에서만 우연히 동작 중. 이번 리팩토링에서 자연 해소한다(§5.4-①).
- **`.gitignore`의 `data/app.db`는 루트 앵커 패턴**: `database.py`를 하위 패키지로 옮기면서 기본 경로를 재정의하지 않으면 `app/core/data/app.db`가 생기고 **gitignore에 안 걸려 로컬 DB가 커밋될 수 있다**.

### 1.5 다행인 점 (조사로 확인됨)

- 문자열 기반 `patch("module.path")`가 코드베이스에 **하나도 없다.** mock은 `app.state.openai_client` 직접 대입과 `app.dependency_overrides[get_db]`(심볼 identity 기반)뿐 → 모듈 경로 변경으로 조용히 깨지는 mock이 없다.
- 프로덕션 코드에 파일 경로 하드코딩이 없다(업로드는 전부 in-memory `UploadFile`). 경로 의존은 `database.py` 기본 경로와 테스트 픽스처(`test_evaluator.py`의 `Data/`)뿐.
- `evaluator/metrics/` 내부는 이미 상대 import(`from .common import ...`) → 패키지째 옮기면 **한 글자도 수정 불필요**.
- 테스트 62개 전부 통과 상태(main 기준), CI·배포 파이프라인 가동 검증 완료.

---

## 2. 목표 구조

### 2.1 최종 디렉토리 트리

```
Capstone_Back/
├── main.py                  # ★ PR1~PR2 사이 한시적 shim: from app.main import app
├── app/
│   ├── __init__.py          # 반드시 빈 파일 (§5.5 — 재수출 금지)
│   ├── main.py              # FastAPI 진입점
│   ├── core/                # 공유 계약 + 인프라
│   │   ├── __init__.py
│   │   ├── schemas.py       # ← schemas.py (+ TC_REQUIREMENTS 편입)
│   │   └── database.py      # ← database.py
│   ├── analysis/            # 도메인 1: 분석 (파싱 + LLM 컬럼 매핑 + 매핑 검증)
│   │   ├── __init__.py
│   │   ├── router.py            # ← routers/analyze.py
│   │   ├── validation_router.py # ← routers/validate.py
│   │   ├── analyzer.py          # ← analyzer.py
│   │   ├── prompt_builder.py    # ← prompt_builder.py
│   │   └── validator.py         # ← validator.py
│   ├── evaluation/          # 도메인 2: 평가 (지표 계산)
│   │   ├── __init__.py
│   │   ├── router.py            # ← routers/evaluate.py
│   │   ├── engine.py            # ← evaluator/engine.py
│   │   ├── preprocessor.py      # ← evaluator/preprocessor.py
│   │   ├── report.py            # ← evaluator/report.py
│   │   └── metrics/             # ← evaluator/metrics/ (내용 무변경)
│   ├── narrative/           # 도메인 3: 서술 (LLM 성적서 7·8·9절)
│   │   ├── __init__.py
│   │   ├── router.py            # ← routers/narrative.py
│   │   ├── narrator.py          # ← narrator.py
│   │   ├── prompt.py            # ← narrative_prompt.py
│   │   ├── fallback.py          # ← narrative_fallback.py
│   │   └── baselines.py         # ← benchmark_baselines.py (프로덕션 모듈임 — 이름에 속지 말 것)
│   └── issuance/            # 도메인 4: 발급 (성적서 발급 메타, DB)
│       ├── __init__.py
│       ├── router.py            # ← routers/reports.py
│       ├── models.py            # ← models.py
│       └── service.py           # ← services/issuance.py
├── tests/
│   ├── conftest.py          # ← conftest.py (내용 무변경)
│   ├── data/                # ← Data/ 의 CSV·JSON (소문자 정규화)
│   │   ├── binary/  multiclass/  multilabel/
│   ├── test_analyze_router.py  test_evaluator.py  test_issuance.py
│   ├── test_narrator.py  test_validator.py  test_route_contract.py
├── scripts/
│   └── llm_smoke_analyze.py # ← test_analyze.py (pytest 아님 — 수동 LLM 스모크 스크립트)
├── data/                    # 런타임 SQLite (gitignore, init_db가 자동 생성)
├── docs/  pytest.ini  render.yaml  requirements*.txt  .env*  (제자리)
```

### 2.2 import 규칙

- 패키지 간에는 **절대 import**: `from app.analysis.analyzer import parse_file_content`
- 같은 패키지 내부의 기존 상대 import(`evaluation/metrics/` 내부)는 그대로 유지
- **의존 방향 (역방향 금지)**: `core ← analysis ← evaluation`, `core ← narrative`, `core ← issuance`
  - `evaluation/router.py`가 `app.analysis.analyzer`(파일 파싱)·`app.analysis.validator`(충돌 검사)를 쓰는 것은 "평가는 상류(분석)를 import할 수 있다" 규칙 안에서 허용

### 2.3 유일한 구조적 결합 해소: `_TC_REQUIREMENTS`

현재 `evaluator/engine.py`가 `from validator import _TC_REQUIREMENTS`로 **밑줄 비공개 이름을 패키지 밖에서** 참조한다. 이 값은 `TaskType`/`ColumnRole`에만 의존하는 순수 데이터 리터럴(validator.py:32~)이므로:

- `app/core/schemas.py`로 옮기고 **`TC_REQUIREMENTS`로 공개 개명**
- 소비자 2곳 수정: `app/analysis/validator.py`(내부 사용 1곳), `app/evaluation/engine.py`(import + 사용 1곳)

---

## 3. 사전 준비 (모든 PR 공통)

```bash
cd ~/Capstone/Capstone_Back

# 1) 원격 최신화 — 로컬 체크아웃이 오래됐을 수 있음 (예: feature/issuance-db는 main보다 뒤)
git fetch origin

# 2) 작업 브랜치는 반드시 최신 origin/main에서 분기
git switch -c refactor/<단계이름> origin/main

# 3) 캐시 청소 — 옮기기 전 모듈의 stale __pycache__가 남으면 로컬에서 가짜 성공/실패가 남
find . -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf .pytest_cache

# 4) 기준선 확인
pytest -q          # 전부 통과해야 시작
```

브랜치 → PR 흐름은 팀 관례대로 (feature 브랜치 → dev → main 승격). **main 승격 = 배포 이벤트**라는 점만 기억할 것.

---

## 4. PR0 — CI 안전망 구축 (코드 이동 0건)

이 PR은 리팩토링 코드가 한 줄도 없다. 목적은 **리팩토링 중 실수를 Render 배포 시점이 아니라 CI 시점에 잡는 것**.

### 4.1 CI에 부팅 검증 스텝 추가

현재 CI는 `pytest -q`만 돌리고 **서버 부팅은 검증하지 않는다**. pytest는 루트가 sys.path에 있어 import 문제를 못 잡는 경우가 있고, 특히 `uvicorn main:app`이라는 **문자열 엔트리포인트 자체**는 아무도 검증하지 않는다.

`.github/workflows/ci.yml`의 `test` job, `pytest` 스텝 뒤에 추가:

```yaml
      - name: 앱 부팅 검증 (render.yaml startCommand와 문자 그대로 동일하게 유지할 것)
        env:
          DATABASE_URL: sqlite:///./ci-boot.db
        run: |
          uvicorn main:app --host 0.0.0.0 --port 8000 &
          UV_PID=$!
          ok=0
          for i in $(seq 1 20); do
            if curl -fsS http://localhost:8000/health; then ok=1; break; fi
            sleep 1
          done
          [ "$ok" = "1" ] || { kill $UV_PID; echo "::error::/health 응답 실패"; exit 1; }
          curl -fsS http://localhost:8000/api/organization   # DB init+seed 왕복 스모크
          kill $UV_PID
```

**규율**: 이 스텝의 uvicorn 커맨드는 render.yaml `startCommand`와 **항상 문자 그대로 동기화**한다(호스트/포트 제외). PR2에서 startCommand를 바꿀 때 이 스텝도 같은 커밋에서 바꾼다.

### 4.2 API 경로 계약 테스트 추가

프론트가 의존하는 `/api/*` 경로가 리팩토링 중 하나라도 사라지거나 바뀌면 테스트가 깨지도록 고정한다.

1. 현재 경로 스냅샷 생성:

```bash
python -c "import json, main; print(json.dumps(sorted(main.app.openapi()['paths'].keys()), ensure_ascii=False, indent=2))"
```

2. 출력된 목록을 붙여넣어 루트에 `test_route_contract.py` 작성 (PR1에서 tests/로 함께 이동한다):

```python
"""API 경로 계약 고정 — 리팩토링 전 구간 동안 /api/* 불변을 기계적으로 보증.

경로를 의도적으로 추가/변경할 때만 이 스냅샷을 갱신할 것.
"""
from main import app   # PR1 이후: from app.main import app

EXPECTED_PATHS = {
    # ← 위 명령의 출력 전체를 붙여넣기 (예: "/health", "/api/analyze-columns", ...)
}

def test_api_paths_unchanged():
    assert set(app.openapi()["paths"].keys()) == EXPECTED_PATHS
```

### 4.3 T0 — Render 설정 출처 확정 (코드 아님, PR0과 병행)

Render 대시보드(hellinii 계정) → `capstone-back` 서비스 → Settings에서 확인·기록:

- [ ] **Start Command 필드가 대시보드에서 편집 가능한 값인가, "Managed by render.yaml"(Blueprint) 표시인가?** → PR2의 Case A/B를 결정하는 정보
- [ ] Deploy Hook URL이 유효한가 (최근 배포 이력에 "via Deploy Hook" 존재)
- [ ] Health Check Path가 `/health`로 설정돼 있는가

### 4.4 PR0 완료 조건 (DoD)

- [ ] CI 녹색 (pytest + 신규 부팅 스텝)
- [ ] main 승격 → deploy job 성공 → Render 배포 live, `/health` 200
- [ ] 이 배포는 동작 무변경이므로 **파이프라인이 살아있음을 확인하는 베이스라인 배포**를 겸한다
- [ ] T0 결과 기록됨

---

## 5. PR1 — 전체 이동 + 루트 shim (본체)

### 5.0 커밋 분리 규율 (중요)

**이동(`git mv`)과 내용 수정을 절대 같은 커밋에 섞지 않는다.** 섞으면 git rename 감지가 깨져서 (a) GitHub 리뷰 화면에서 "파일 전체 삭제 + 신규 추가"로 보이고, (b) `git log --follow` 히스토리 추적이 약해진다.

- **commit 1**: `git mv`만 — 내용 변경 0바이트. 리뷰어는 rename 목록만 훑으면 됨
- **commit 2**: 내용 수정만 — import 재작성 + 의도적 수정 4건 + 테스트 경로 수정 + pytest.ini. 리뷰어는 이 커밋만 라인 리뷰
- **commit 3**: 루트 shim 신설 (신규 파일 1개)

PR 본문에 이 안내를 명시할 것: *"commit 1은 rename 목록 확인만, commit 2·3만 라인 리뷰해주세요."*

> 중간 커밋 단독으로는 CI가 깨진다(이동만 하고 import를 안 고친 상태). CI는 PR HEAD만 검사하므로 문제없다.

### 5.1 commit 1 — 순수 이동

```bash
mkdir -p app/core app/analysis app/evaluation app/narrative app/issuance tests scripts

# 진입점 · 공용
git mv main.py       app/main.py
git mv schemas.py    app/core/schemas.py
git mv database.py   app/core/database.py

# analysis (분석: 파싱 + 컬럼 매핑 + 매핑 검증)
git mv analyzer.py          app/analysis/analyzer.py
git mv prompt_builder.py    app/analysis/prompt_builder.py
git mv validator.py         app/analysis/validator.py
git mv routers/analyze.py   app/analysis/router.py
git mv routers/validate.py  app/analysis/validation_router.py

# evaluation (평가: 지표 계산)
git mv evaluator/engine.py        app/evaluation/engine.py
git mv evaluator/preprocessor.py  app/evaluation/preprocessor.py
git mv evaluator/report.py        app/evaluation/report.py
git mv evaluator/metrics          app/evaluation/metrics
git mv routers/evaluate.py        app/evaluation/router.py

# narrative (서술)
git mv narrator.py            app/narrative/narrator.py
git mv narrative_prompt.py    app/narrative/prompt.py
git mv narrative_fallback.py  app/narrative/fallback.py
git mv benchmark_baselines.py app/narrative/baselines.py
git mv routers/narrative.py   app/narrative/router.py

# issuance (발급)
git mv models.py            app/issuance/models.py
git mv services/issuance.py app/issuance/service.py
git mv routers/reports.py   app/issuance/router.py

# 껍데기 __init__ 삭제 (evaluator/__init__.py의 재수출 `from .engine import evaluate`는 소비자 없음 — 확인됨)
git rm routers/__init__.py services/__init__.py evaluator/__init__.py

# tests
git mv conftest.py             tests/conftest.py
git mv test_analyze_router.py  tests/test_analyze_router.py
git mv test_evaluator.py       tests/test_evaluator.py
git mv test_issuance.py        tests/test_issuance.py
git mv test_narrator.py        tests/test_narrator.py
git mv test_validator.py       tests/test_validator.py
git mv test_route_contract.py  tests/test_route_contract.py   # PR0에서 만든 것

# 테스트 픽스처 (대문자 Data → 소문자 tests/data 정규화)
mkdir -p tests/data
git mv Data/Binary     tests/data/binary
git mv Data/MultiClass tests/data/multiclass
git mv Data/MultiLabel tests/data/multilabel
rm -f Data/app.db && rmdir Data   # app.db는 untracked 런타임 파일 — 삭제해도 무방(재생성됨)

# 수동 스크립트 (test_* 패턴에서 벗어나 pytest 수집 원천 차단)
git mv test_analyze.py scripts/llm_smoke_analyze.py

# 빈 __init__.py 생성 (내용은 §5.5 참조 — docstring만)
touch app/__init__.py app/core/__init__.py app/analysis/__init__.py \
      app/evaluation/__init__.py app/narrative/__init__.py app/issuance/__init__.py
git add app/

git commit -m "refactor: 파일 이동만 — 도메인 패키지 구조로 재배치 (내용 무변경)"
```

### 5.2 commit 2 — import 재작성 + 의도적 수정 4건

#### 5.2.1 파일별 import 수정 목록

아래는 **각 파일에서 고칠 import 라인 전체**다. 나열되지 않은 파일(`app/narrative/prompt.py`, `app/narrative/baselines.py`, `app/evaluation/preprocessor.py`, `app/evaluation/report.py`, `app/evaluation/metrics/*`, `tests/conftest.py`)은 **수정할 것이 없다.**

**`app/main.py`**
```python
# 변경 전
from database import DATABASE_URL, init_db, seed_organization
from routers import analyze, evaluate, validate, narrative, reports
# 변경 후 (load_dotenv() 호출이 이 import들보다 앞이라는 기존 순서·주석은 그대로 유지)
from app.core.database import DATABASE_URL, init_db, seed_organization
from app.analysis.router import router as analyze_router
from app.analysis.validation_router import router as validate_router
from app.evaluation.router import router as evaluate_router
from app.narrative.router import router as narrative_router
from app.issuance.router import router as reports_router
```
```python
# 라우터 등록부 변경 전 → 후
app.include_router(analyze.router)    →  app.include_router(analyze_router)
app.include_router(evaluate.router)   →  app.include_router(evaluate_router)
app.include_router(validate.router)   →  app.include_router(validate_router)
app.include_router(narrative.router)  →  app.include_router(narrative_router)
app.include_router(reports.router)    →  app.include_router(reports_router)
```
```python
# __main__ 블록
uvicorn.run("main:app", ...)  →  uvicorn.run("app.main:app", ...)
```

**`app/analysis/analyzer.py`**
```python
from schemas import (...)         →  from app.core.schemas import (...)
from prompt_builder import (...)  →  from app.analysis.prompt_builder import (...)
```

**`app/analysis/validator.py`** — §5.2.2-②의 TC_REQUIREMENTS 이동과 함께
```python
from schemas import (...)  →  from app.core.schemas import (..., TC_REQUIREMENTS)
```

**`app/analysis/prompt_builder.py`**
```python
from schemas import TaskType, VALID_ROLES_BY_TASK  →  from app.core.schemas import TaskType, VALID_ROLES_BY_TASK
```

**`app/analysis/router.py`** (구 routers/analyze.py)
```python
from schemas import (...)     →  from app.core.schemas import (...)
from analyzer import (...)    →  from app.analysis.analyzer import (...)
from validator import validate_mapping  →  from app.analysis.validator import validate_mapping
```

**`app/analysis/validation_router.py`** (구 routers/validate.py)
```python
from schemas import (...)                     →  from app.core.schemas import (...)
from analyzer import parse_file_content       →  from app.analysis.analyzer import parse_file_content
from validator import find_column_conflicts   →  from app.analysis.validator import find_column_conflicts
```

**`app/evaluation/router.py`** (구 routers/evaluate.py)
```python
from schemas import EvaluateRequest, EvaluateResponse   →  from app.core.schemas import ...
from analyzer import parse_file_content                 →  from app.analysis.analyzer import parse_file_content
from evaluator.engine import evaluate as run_evaluation →  from app.evaluation.engine import evaluate as run_evaluation
from evaluator.report import generate_report            →  from app.evaluation.report import generate_report
from validator import find_column_conflicts             →  from app.analysis.validator import find_column_conflicts
```

**`app/evaluation/engine.py`**
```python
from validator import _TC_REQUIREMENTS  →  from app.core.schemas import TC_REQUIREMENTS
# 본문 사용처(약 11행)의 _TC_REQUIREMENTS 도 TC_REQUIREMENTS 로 개명
# `from .metrics import ...`, `from .preprocessor import ...` 는 무변경
```

**`app/narrative/narrator.py`**
```python
from schemas import (...)                           →  from app.core.schemas import (...)
from benchmark_baselines import build_benchmark_refs →  from app.narrative.baselines import build_benchmark_refs
from narrative_fallback import build_fallback_narrative →  from app.narrative.fallback import build_fallback_narrative
from narrative_prompt import build_system_prompt, build_user_prompt, build_response_schema
                                                    →  from app.narrative.prompt import build_system_prompt, build_user_prompt, build_response_schema
```

**`app/narrative/fallback.py`**
```python
from schemas import (...)  →  from app.core.schemas import (...)
```

**`app/narrative/router.py`** (구 routers/narrative.py)
```python
from schemas import NarrativeRequest, NarrativeResponse  →  from app.core.schemas import ...
from narrator import generate_narrative                  →  from app.narrative.narrator import generate_narrative
```

**`app/issuance/models.py`**
```python
from database import Base  →  from app.core.database import Base
```

**`app/issuance/service.py`** (구 services/issuance.py)
```python
from models import Issuance, Organization, Report  →  from app.issuance.models import Issuance, Organization, Report
```

**`app/issuance/router.py`** (구 routers/reports.py)
```python
from database import get_db                    →  from app.core.database import get_db
from models import Organization, Report        →  from app.issuance.models import Organization, Report
from schemas import (...)                      →  from app.core.schemas import (...)
from services import issuance as issuance_service  →  from app.issuance import service as issuance_service
from services.issuance import IssuanceError    →  from app.issuance.service import IssuanceError
```

**테스트 파일들** (`tests/`)
```python
# tests/test_analyze_router.py
from analyzer import _reconcile_llm_columns  →  from app.analysis.analyzer import _reconcile_llm_columns
from schemas import ColumnRole               →  from app.core.schemas import ColumnRole
import main → main.app.state...              →  from app.main import app → app.state...   # 함수 내부 4곳

# tests/test_evaluator.py
from evaluator.engine import evaluate          →  from app.evaluation.engine import evaluate
from evaluator.preprocessor import preprocess_data  →  from app.evaluation.preprocessor import preprocess_data
# + 픽스처 경로: §5.2.2-④

# tests/test_issuance.py
import database / from database import (...)   →  from app.core import database / from app.core.database import (...)
from models import (...)                       →  from app.issuance.models import (...)
from services import issuance as svc           →  from app.issuance import service as svc
from services.issuance import IssuanceError    →  from app.issuance.service import IssuanceError
from main import app                           →  from app.main import app

# tests/test_narrator.py
from schemas import (...)             →  from app.core.schemas import (...)
from narrator import (...)            →  from app.narrative.narrator import (...)
from narrative_fallback import (...)  →  from app.narrative.fallback import (...)
from benchmark_baselines import (...) →  from app.narrative.baselines import (...)

# tests/test_validator.py
from schemas import (...)                    →  from app.core.schemas import (...)
from validator import (...)                  →  from app.analysis.validator import (...)
from evaluator.preprocessor import (...)     →  from app.evaluation.preprocessor import (...)
from evaluator.engine import evaluate as run_evaluation  →  from app.evaluation.engine import evaluate as run_evaluation

# tests/test_route_contract.py
from main import app  →  from app.main import app
```

**`scripts/llm_smoke_analyze.py`** (구 test_analyze.py)
```python
# sys.path 조작: 파일 위치가 아니라 레포 루트를 가리키도록
sys.path.insert(0, os.path.dirname(__file__))
  →  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import TaskType        →  from app.core.schemas import TaskType
from analyzer import (...)          →  from app.analysis.analyzer import (...)

# 데이터 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
  →  DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")
```

#### 5.2.2 의도적 수정 4건 (import 치환이 아닌 실제 코드 변경)

**① `app/core/database.py` — 기본 DB 경로를 레포 루트로 재앵커** (14행)

```python
# 변경 전 — database.py가 루트에 있을 때만 <루트>/data/app.db 를 가리킴
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "app.db")

# 변경 후 — app/core/ 로 이동해도 <레포 루트>/data/app.db 유지
#   (__file__ = app/core/database.py → 상위 2단계가 레포 루트)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "app.db")
```

`init_db()`가 이미 `os.makedirs(...)`로 디렉토리를 생성하므로(96~99행) 추가 조치 불필요. `.gitignore`의 `data/app.db` 패턴과 정확히 일치하고, 소문자 `data/`라 대소문자 문제도 함께 해소된다.

**② `_TC_REQUIREMENTS` → `app/core/schemas.py`의 `TC_REQUIREMENTS`로 이동·공개 개명**

- `app/analysis/validator.py`에서 `_TC_REQUIREMENTS: dict[TaskType, dict[str, set[ColumnRole]]] = { ... }` 리터럴 전체(32행~)를 **잘라내어** `app/core/schemas.py`의 `VALID_ROLES_BY_TASK` 정의 아래에 붙여넣고 이름을 `TC_REQUIREMENTS`로 변경 (주석 포함 그대로).
- `app/analysis/validator.py`: 사용처 1곳(약 246행) `_TC_REQUIREMENTS[task_type]` → `TC_REQUIREMENTS[task_type]`, import에 `TC_REQUIREMENTS` 추가.
- `app/evaluation/engine.py`: §5.2.1 참조.

**③ `app/core/database.py` — 지연 import 경로 갱신 (함수 내부 위치는 절대 유지)**

```python
# init_db() 내부 (101행)
import models  # noqa: F401
  →  import app.issuance.models  # noqa: F401

# seed_organization() 내부 (107행)
from models import Organization
  →  from app.issuance.models import Organization
```

> ⚠️ 이 두 줄을 파일 최상단으로 끌어올리면 `models.py ↔ database.py` 순환 import로 깨진다. 함수 내부에 그대로 둘 것 (기존 주석 "순환 방지"도 유지).

**④ `tests/test_evaluator.py` — 픽스처 경로** (9~12행)

```python
# 변경 전 (루트 기준 대문자 Data)
BASE_DIR = Path(__file__).parent
BINARY_CSV = BASE_DIR / "Data" / "Binary" / "binary_test_data_200.csv"
...

# 변경 후 (tests/ 기준 소문자 data — commit 1에서 tests/data/binary 등으로 이동됨)
BASE_DIR = Path(__file__).parent
BINARY_CSV = BASE_DIR / "data" / "binary" / "binary_test_data_200.csv"
...  # multiclass, multilabel 동일 패턴
```

#### 5.2.3 `pytest.ini` 갱신

conftest.py가 tests/로 내려가면 "루트 conftest가 sys.path에 루트를 넣어주던" 암묵적 보장이 사라진다. 선언적으로 대체한다:

```ini
[pytest]
asyncio_mode = auto
pythonpath = .
testpaths = tests
```

- `pythonpath = .` (pytest≥7, 현재 pytest≥8 핀): 루트를 sys.path에 삽입 → `from app...` 절대 import 보장
- `testpaths = tests`: scripts/ 등 오수집 차단
- pytest.ini 자체는 루트에 남아 rootdir 앵커 유지

```bash
git commit -am "refactor: 도메인 패키지 import 경로 재작성 + DB 경로 재앵커 + TC_REQUIREMENTS 공개 이동"
```

#### 5.2.4 `app/__init__.py` 계열 — docstring만 (§5.5의 이유)

각 `__init__.py`는 빈 파일 또는 docstring만. 특히 `app/__init__.py`에는 이유를 명시해두면 후임 실수를 막는다:

```python
"""빈 파일로 유지할 것 — 어떤 재수출도 추가 금지.

여기에 `from app.main import app` 같은 편의 재수출을 넣으면 `import app` 시점에
load_dotenv()보다 먼저 app.core.database가 로드되어, 로컬 .env의 DATABASE_URL이
무시된 채 SQLite 기본값으로 조용히 폴백한다 (app/main.py의 import 순서 주석 참조).
"""
```

### 5.3 commit 3 — 루트 shim 신설

루트에 새 `main.py` 생성:

```python
"""Render startCommand(`uvicorn main:app`) 하위 호환 shim — 실체는 app/main.py.

startCommand를 `uvicorn app.main:app`으로 전환하고 배포 1사이클 검증(PR2)을
마치기 전까지 삭제 금지.
"""
from app.main import app  # noqa: F401
```

이로써 `uvicorn main:app`(Render의 startCommand, CI 부팅 스텝)이 **그대로 동작**한다. PR1은 render.yaml·대시보드·Deploy Hook 어느 것도 건드리지 않는다 — **배포 설정 접점 0**.

```bash
git add main.py && git commit -m "refactor: 루트 main.py를 하위 호환 shim으로 신설 (uvicorn main:app 유지)"
```

### 5.4 PR1 로컬 검증 (머지 전 필수)

```bash
# 0) 캐시 청소 (필수 — 옮기기 전 모듈의 stale pyc가 가짜 성공을 만든다)
find . -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf .pytest_cache

# 1) 잔여 flat import 검출 — 결과가 0줄이어야 함
grep -rnE "^(from|import) (schemas|analyzer|validator|prompt_builder|narrator|narrative_prompt|narrative_fallback|benchmark_baselines|database|models|evaluator|services|routers)\b" \
  --include="*.py" app/ tests/ scripts/

# 2) 테스트 전건 (62개 + 계약 테스트)
pytest -q

# 3) 부팅 — Render가 실행할 바로 그 문자열로
uvicorn main:app --port 8000
#    시작 로그 2종 확인: "✅ 발급 메타 DB 초기화 완료", OpenAI 클라이언트 로그(또는 ⚠️ 폴백 경고)

# 4) 스모크 (다른 터미널)
curl -s localhost:8000/health                      # {"status":"ok"}
curl -s localhost:8000/api/organization            # 시드 기관 반환 (DB 왕복)
curl -s localhost:8000/openapi.json | python -c "import sys,json; print(sorted(json.load(sys.stdin)['paths'].keys()))"
#    → 리팩토링 전 스냅샷과 diff 0

# 5) .env 없는 셸에서도 부팅되는지 1회 (조기 import 회귀 감지)
env -i PATH="$PATH" uvicorn main:app --port 8001   # 부팅 후 Ctrl-C

# 6) 히스토리 보존 확인 (표본 3개)
git log --follow --oneline app/analysis/validator.py | head -3
git log --follow --oneline app/narrative/narrator.py | head -3
git log --follow --oneline app/issuance/service.py | head -3

# 7) 루트에 생성된 data/app.db가 git에 안 잡히는지
git status --short   # data/ 관련 항목이 없어야 함
```

### 5.5 PR1 완료 조건 (DoD)

- [ ] 위 로컬 검증 7항목 전부 통과
- [ ] CI 녹색 — **Linux 대소문자 검증은 CI가 유일한 게이트** (macOS 로컬 통과는 불충분)
- [ ] main 승격 → 배포 후 체크리스트(§8) 전부 통과
- [ ] Render 로그에서 shim 경유 정상 부팅 확인

---

## 6. PR2 — startCommand 전환 + shim 제거

T0(§4.3) 결과에 따라 분기한다. **어느 경우든 shim이 있는 동안은 실패해도 서비스가 깨지지 않는다.**

### Case A — Blueprint(render.yaml) 관리로 확인된 경우

1. 한 커밋에서 동시 변경 (문자열 동기화 규율):
   - `render.yaml`: `startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - `.github/workflows/ci.yml` 부팅 스텝: `uvicorn app.main:app --host 0.0.0.0 --port 8000 &`
2. main 승격 → 배포 → **Render 이벤트/로그에서 실제 실행된 커맨드 문자열을 육안 확인**
3. 반영이 안 됐다면 = 사실상 대시보드 수동 관리였다는 뜻 → shim 덕에 서비스는 정상이므로 Case B로 전환 (무해)

### Case B — 대시보드 수동 관리로 확인된 경우

1. Render 대시보드에서 Start Command를 `uvicorn app.main:app --host 0.0.0.0 --port $PORT`로 먼저 변경
2. Manual Deploy(또는 빈 커밋 main 승격)로 배포 → 로그에서 신규 커맨드 부팅 확인
3. 이후 render.yaml + CI 부팅 스텝도 같은 문자열로 맞추는 PR (문서 정합성)

### shim 제거

신규 커맨드로 **성공 배포 1사이클을 확인한 뒤**, 루트 `main.py`(shim) 삭제 커밋/PR. CI 부팅 스텝이 이미 `app.main:app`이므로 삭제 실수는 CI가 잡는다.

**Fallback**: T0에서 설정 출처를 확정하지 못하면 **shim 영구 유지**로 전환한다. 3줄짜리 파일 하나가 비용의 전부이고, 두 진입점 공존의 혼동은 shim의 docstring이 방어한다.

### PR2 완료 조건 (DoD)

- [ ] Render 로그의 실행 커맨드 = `uvicorn app.main:app ...`
- [ ] `/health` 200, `/api/organization` 정상
- [ ] 프론트(Vercel)에서 대표 플로우 1회: 업로드 → 컬럼 분석 → 평가 → 발급
- [ ] shim 삭제 후 배포도 정상

---

## 7. PR3 (후속) — schemas.py 도메인별 분리

구조가 안정된 후(PR2 완료 + 정상 운영 확인) 별도 진행. 순수 Python 변경이라 배포 리스크는 없고 CI가 전부 검증한다.

`app/core/schemas.py`는 이미 도메인 섹션 주석(`── Step 1 ──` 등)으로 구획돼 있어 분배는 기계적이다:

| 대상 파일 | 옮길 심볼 |
|---|---|
| `app/core/schemas.py` (잔류 — 공유 계약) | `TaskType`, `ReportPurpose`, `ColumnRole`, `VALID_ROLES_BY_TASK`, `TC_REQUIREMENTS`, **`ColumnMapping`, `DataMetadata`** |
| `app/analysis/schemas.py` | `ColumnMatchNote`, `AnalysisResponse`, `ConfirmMappingRequest/Response`, `MappingValidationError/Warning`, `ValidationCheckItem`, `ExecutionSummaryItem`, `ValidateDataResponse` |
| `app/evaluation/schemas.py` | `EvaluateRequest`, `EvaluateResponse` |
| `app/narrative/schemas.py` | `MetricFact`, `PerClassFact`, `ConfusionFact`, `DistributionFact`, `LatencyFact`, `FactSheet`, `NarrativeRequest/Response`, `InterpretationOut`, `ConclusionOut`, `RecommendationNarrativeOut`, `RecommendationOut`, `GroundingInfo`, `NarrativeMeta` |
| `app/issuance/schemas.py` | `OrganizationIn/Out`, `IssueRequest`, `ReissueRequest`, `IssuanceHistoryItem`, `IssuanceOut` |

**핵심 설계 원칙**: `ColumnMapping`/`DataMetadata`는 분석(Step 1)의 산출물이자 평가(Step 3)의 입력인 **파이프라인 인계 계약**이므로 core에 남긴다. 그래야 `evaluation/schemas.py → analysis/schemas.py` 교차 import가 생기지 않고, **모든 도메인 스키마가 core만 바라보는 별(star) 구조**가 되어 순환이 원천 불가능해진다.

분리 후 검증: `grep -rn "from app.core.schemas import" app/ tests/`로 각 파일이 실제 공유 심볼만 core에서 가져오는지 확인 + `pytest -q`.

---

## 8. 배포 후 체크리스트 (모든 main 승격 공통, ~5분)

1. [ ] GitHub Actions **deploy job 성공** (포크 미러 push + Deploy Hook 200)
2. [ ] Render **Events**: 신규 배포 live, 헬스체크 통과 / **Logs**: 부팅 로그 정상("✅ 발급 메타 DB 초기화", backend=postgresql), 트레이스백 없음
3. [ ] `curl https://capstone-back-59z8.onrender.com/health` → `{"status":"ok"}` (free tier 콜드스타트로 첫 응답 ~1분 지연 가능 — 정상)
4. [ ] `curl https://capstone-back-59z8.onrender.com/api/organization` → 기관 정보 (Neon 왕복 확인)
5. [ ] 프론트 `https://capstone-beta-lilac.vercel.app`에서 대표 플로우 1회: 업로드 → 컬럼 분석 → 평가 → 발급

---

## 9. 롤백 절차

- **부팅 실패류**: Render 헬스체크가 실패하면 **이전 버전이 그대로 유지**된다(무정지). 급하지 않게 아래 표준 절차로 수습.
- **표준 절차** (부팅은 되는데 기능이 깨진 경우 포함):
  ```bash
  git switch main && git pull
  git revert -m 1 <문제가 된 머지커밋 SHA>
  # → PR 또는 직접 push → CI 통과 → 자동으로 미러+Deploy Hook → 이전 코드 재배포 (소요 ≈ 5~10분)
  ```
  순수 코드 이동이라 **DB 롤백은 항상 불필요** (스키마 무변경, `create_all`은 additive).
- **비상 최단 경로** (CI 파이프라인 자체가 죽었을 때만): 포크(hellinii) main을 직전 정상 커밋으로 force-push + Deploy Hook 수동 curl. 팀 main과 포크가 일시적으로 어긋나지만 다음 정상 배포의 미러 force-push가 자가 치유한다. **최후 수단으로만.**
- Render free 플랜에는 대시보드 롤백 버튼이 없다 — **git revert가 유일한 정규 경로**.

---

## 10. 이번에 하지 않는 것 (별도 이슈로)

| 항목 | 이유 |
|---|---|
| `pyproject.toml` / 패키지 설치 도입 | uvicorn의 cwd sys.path 삽입 + pytest `pythonpath = .`로 충분. 빌드/배포 커맨드까지 건드리는 별개 리스크 — 이동 PR과 섞으면 사고 시 원인 분리가 안 됨 |
| Alembic (마이그레이션) | 스키마 변경이 없는 리팩토링과 무관 |
| CORS `allow_origins=["*"]` 제한 | 정책 변경 — 순수 이동 PR과 혼합 금지 |
| isort / ruff 도입 | **`main.py`의 load_dotenv 순서를 깨뜨릴 수 있음** — 도입 시 해당 블록에 `# isort: skip` 가드 필요. 이슈에 이 경고를 반드시 기록할 것 |
| 루트의 `9adc...백엔드_ToDo.pdf` | 원하면 docs/로 옮기는 정도만 (선택) |

---

## 11. 트러블슈팅 — 증상 → 원인 → 조치

| 증상 | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError: No module named 'schemas'` (등 옛 이름) | flat import 잔존 | §5.4-1의 grep으로 위치 특정 후 치환표(§5.2.1) 적용 |
| 로컬에서 `.env`의 DATABASE_URL이 무시되고 SQLite로 붙음 | `__init__.py`에 재수출을 넣어 dotenv보다 먼저 database가 import됨 | `app/__init__.py`·`app/core/__init__.py`를 빈 파일로 되돌리기 (§5.2.4) |
| `app/core/data/app.db` 파일이 생기고 `git status`에 잡힘 | `_DEFAULT_DB_PATH` 재앵커 누락 | §5.2.2-① 적용, 생성된 파일 삭제 |
| 로컬(macOS)은 통과하는데 CI(Linux)에서만 test_evaluator 실패 (`FileNotFoundError`) | 픽스처 경로 대소문자 불일치 (`Data` vs `data`) | §5.2.2-④ 확인 — 경로 전부 소문자 `data/` |
| `pytest`가 테스트를 0개 수집하거나 import error | `pytest.ini`에 `pythonpath = .`/`testpaths = tests` 누락, 또는 stale `__pycache__` | §5.2.3 적용 + 캐시 청소 |
| 부팅 시 순환 import 에러 (`partially initialized module`) | database.py의 지연 import를 최상단으로 옮김 | §5.2.2-③ — 함수 내부로 되돌리기 |
| render.yaml의 startCommand를 바꿨는데 Render에 반영 안 됨 | 서비스가 대시보드 수동 생성 (Blueprint 아님) | §6 Case B 절차로 전환 — shim 덕에 서비스는 정상 |
| GitHub 리뷰 화면에서 이동 파일이 "삭제+신규"로 보임 | mv와 내용 수정을 같은 커밋에 섞음 | 커밋 분리 규율(§5.0) — `git reset`으로 커밋 재구성 |
| 배포 후 특정 API만 404 | 라우터 등록 누락 (main.py의 include_router 5개 확인) | 계약 테스트(§4.2)가 CI에서 먼저 잡아줌 — EXPECTED_PATHS와 대조 |
| deploy job에서 포크 push 403 | (기존 이슈) checkout의 `persist-credentials: false` 누락 또는 PAT 권한 | docs/DEPLOYMENT_PLAN.md 및 CI 주석 참조 — 리팩토링과 무관 |

---

## 부록 A. 왜 이런 순서인가 (설계 근거 요약)

- **도메인별 점진 이동(파일 몇 개씩 여러 PR)을 기각한 이유**: flat import와 패키지 import가 공존하는 기간에 같은 모듈이 두 경로로 import되면 **모듈 이중 인스턴스**(module-level 상태 분열)라는 가장 잡기 어려운 버그가 생긴다. 이 규모(~30파일, import ~40줄)는 한 PR로 리뷰 가능하므로 한 번에 옮기되, 배포에 닿는 변경(엔트리포인트 전환)만 분리했다.
- **shim을 두는 이유**: 배포 설정(startCommand)이 repo 밖(대시보드)에 있을 가능성 때문에, "코드 이동"과 "배포 설정 변경"을 시간적으로 분리했다. shim이 있는 동안 어느 쪽이 실패해도 다른 쪽이 방어한다.
- **`TC_REQUIREMENTS`를 core로 올린 이유**: 순수 데이터 리터럴이고 의존 대상이 공유 enum뿐이라 순환이 생길 수 없으며, 진짜 문제였던 "밑줄 비공개 이름의 패키지 경계 밖 참조"가 공개 승격으로 해소된다.
- **절대 import를 선택한 이유**: 기존 flat import에서의 치환이 기계적이고(1:1 매핑, grep 검증 가능), uvicorn·pytest 양쪽에서 동일하게 해석되며, 파일을 도메인 간 재이동할 때 상대 import(`from ..core import`)처럼 깨지지 않는다.
