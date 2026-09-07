# 조직/발급 메타 DB 설계 (P2-11)

> **⚠ 이 문서는 2026-09-07 자로 이력으로 동결되었습니다.**
>
> 여기 적힌 계획·설계는 **이미 구현되어 배포됐습니다.** 문서의 '상태' 표기와 세부
> 서술은 작성 시점(구현 전)의 것이므로 **현재 코드의 근거로 삼지 마십시오.**
> 현행 구조는 `docs/ARCHITECTURE.md` 와 코드가 정본입니다.
>
> 이 문서를 남겨 두는 이유는 **왜 그렇게 만들었는가**를 잃지 않기 위해서입니다.
> 앞으로 갱신하지 않습니다(ISSUES.md H-09, 2026-09-07 ★결정 11).


> 현재 코드 구조는 docs/ARCHITECTURE.md 참조(이 문서는 설계 근거).

> 작성일: 2026-06-24
> 범위: 성적서의 **수행기관(performer)·발급정보(signature)·리포트 번호(reportId)** 를 프론트 하드코딩에서 백엔드 DB로 이관
> 상태: **설계 계획 (구현 전)** — 본 문서 확정 후 구현 착수

---

## 1. 배경 & 문제 정의

현재 백엔드는 **완전 stateless** 다(DB 없음). 성적서의 조직/발급 메타가 프론트에 하드코딩되어 있다:

| 항목 | 현재 위치 | 문제 |
|---|---|---|
| 수행기관(performer) | [`reportConstants.ts`](../../Capstone_Front/src/lib/report/reportConstants.ts) `DEFAULT_PERFORMER` | 코드 상수. 변경하려면 재배포 |
| 발급정보(signature) | [`mapWorkflowToFinalReport.ts`](../../Capstone_Front/src/lib/report/mapWorkflowToFinalReport.ts) `buildSignature()` | issuer/발급일/이력이 매번 즉석 생성 — 추적성 없음 |
| 리포트 번호(reportId) | 같은 파일 `buildReportId()` | `RPT-<year>-<seq>` 의 seq 가 **timestamp 기반 의사난수** → **순번 아님, 충돌 가능, 재현 불가** |

→ 진짜 성적서 번호(연도별 순번)와 발급 이력(재발급 버전업)은 **상태(state)** 가 필요하므로 stateless 로는 불가능하다. DB 로 이관한다.

## 2. 확정 결정 사항

| 항목 | 결정 | 근거 |
|---|---|---|
| DB | **SQLite + SQLAlchemy ORM** | 단일 백엔드·낮은 동시성·채번에 충분. ORM 추상화로 추후 PostgreSQL 전환 용이 |
| 채번 시점 | **명시적 '발급' 시점** | 진짜 성적서 번호의 의미(초안↔발급본 구분). 평가 전/취소된 run 에 번호 낭비 방지 |
| 기관 범위 | **단일 고정 기관** | Capstone 범위. `organization` 테이블 singleton(1행) + 시드 |
| 재발급 | **발급 이력 관리(버전업)** | 성적서 정정 시 같은 번호 유지 + 버전 차수 증가. `signature.history` 가 이미 이 구조를 가정 |

## 3. 데이터 모델 (스키마)

3개 테이블. 정규화: **성적서 헤더(report) 1 : N 발급차수(issuance)**, 기관은 singleton.

```
┌────────────────────────┐        ┌─────────────────────────────┐
│ organization (singleton)│        │ report (성적서 헤더 = 채번 단위) │
│  id (PK, =1)            │◀──┐    │  id (PK)                     │
│  org_name              │   └────│  org_id (FK)                 │
│  department            │        │  report_no (UNIQUE)          │
│  evaluator             │        │  year, seq  (UNIQUE together)│
│  contact               │        │  run_id (평가 연결)           │
│  address               │        │  model_name, model_version   │
│  updated_at            │        │  current_version             │
└────────────────────────┘        │  created_at                  │
                                   └──────────────┬───────────────┘
                                                  │ 1:N
                                   ┌──────────────▼───────────────┐
                                   │ issuance (발급 차수/이력)      │
                                   │  id (PK)                     │
                                   │  report_id (FK)              │
                                   │  version  ("v1.0","v1.1")    │
                                   │  issuer                      │
                                   │  issued_at                   │
                                   │  note                        │
                                   │  status ("issued"/"superseded")│
                                   └──────────────────────────────┘
```

### 3.1 `organization` (수행기관 — singleton)

| 컬럼 | 타입 | 제약 | 설명 |
|---|---|---|---|
| id | INTEGER | PK (항상 1) | singleton |
| org_name | TEXT | NOT NULL | "한국 AI 인증원" |
| department | TEXT | | "평가부" (issuer 조합용) |
| evaluator | TEXT | | "자동 평가 엔진" (performer.evaluator) |
| contact | TEXT | | 연락처 |
| address | TEXT | NULL | 주소(선택) |
| updated_at | DATETIME | | 갱신 시각 |

→ 프론트 `performer` = `{orgName: org_name, evaluator, contact}` 로 매핑.

### 3.2 `report` (성적서 헤더 — 채번 단위)

| 컬럼 | 타입 | 제약 | 설명 |
|---|---|---|---|
| id | INTEGER | PK autoincrement | |
| report_no | TEXT | UNIQUE NOT NULL | "RPT-2026-0001" (채번 결과) |
| year | INTEGER | NOT NULL | 2026 (채번 연도) |
| seq | INTEGER | NOT NULL | 1 (연도 내 순번) |
| run_id | TEXT | NULL, INDEX | 프론트 워크스페이스 run id (어떤 평가의 발급인지) |
| model_name | TEXT | | 대상 모델명(표시/검색용) |
| model_version | TEXT | | 대상 모델 버전 |
| org_id | INTEGER | FK→organization.id | 수행기관 |
| current_version | TEXT | NOT NULL | 최신 발급 버전("v1.1") |
| created_at | DATETIME | NOT NULL | 최초 발급 시각 |
| | | **UNIQUE(year, seq)** | 채번 충돌 이중 방어 |

> `run_id` 로 "이미 발급된 평가인가"를 판별해 재발급 분기. 같은 run 재발급 시 **같은 report_no 유지**.

### 3.3 `issuance` (발급 차수 — 이력)

| 컬럼 | 타입 | 제약 | 설명 |
|---|---|---|---|
| id | INTEGER | PK autoincrement | |
| report_id | INTEGER | FK→report.id, INDEX | 소속 성적서 |
| version | TEXT | NOT NULL | "v1.0", "v1.1" … |
| issuer | TEXT | NOT NULL | "한국 AI 인증원 평가부" |
| issued_at | DATETIME | NOT NULL | 발급 일시 |
| note | TEXT | | "최초 발급" / "정정 발급: …" |
| status | TEXT | NOT NULL | "issued"(현행) / "superseded"(상위 버전에 의해 대체) |

→ 프론트 `signature` = `{issuer: 최신.issuer, signedAt: 최신.issued_at, history: [{version, issuedAt, note} …]}`.

## 4. 채번 로직 (연도별 순번)

발급 요청 시 트랜잭션 내에서:

```
BEGIN IMMEDIATE                       # SQLite 쓰기 잠금(단일 writer)
report = SELECT * FROM report WHERE run_id = :run_id
if report is None:                    # ── 신규 발급 ──
    year = <발급 연도>                 # 서버 시계
    seq  = COALESCE(MAX(seq),0)+1  FROM report WHERE year = :year
    report_no = f"RPT-{year}-{seq:04d}"
    report = INSERT report(report_no, year, seq, run_id, model_*, org_id=1, current_version="v1.0")
    INSERT issuance(report_id, version="v1.0", issuer, issued_at=now, note="최초 발급", status="issued")
else:                                 # ── 재발급(정정) ──
    prev = 최신 issuance(report)
    UPDATE prev.status = "superseded"
    new_version = bump(prev.version)   # v1.0 → v1.1
    UPDATE report.current_version = new_version
    INSERT issuance(report_id, version=new_version, issuer, issued_at=now, note=:note, status="issued")
COMMIT
```

- **동시성**: SQLite 는 쓰기를 직렬화한다. `BEGIN IMMEDIATE` + `UNIQUE(year, seq)` 제약으로 이중 방어. 충돌(IntegrityError) 시 1회 재시도.
- **버전 증가 규칙**: `v1.0 → v1.1 → v1.2` (minor 증가). 표기 함수 `bump("v1.0") = "v1.1"`.
- **연도 경계**: seq 는 연도별로 리셋(2027 년 첫 발급 = RPT-2027-0001).
- 서버 시계는 채번 시점에 사용(스크립트 결정성과 무관 — 런타임 발급 시각).

## 5. API 설계

신규 라우터 [`app/issuance/router.py`](../app/issuance/router.py), prefix `/api`.

| 메서드·경로 | 용도 | 요청 | 응답 |
|---|---|---|---|
| `GET /api/organization` | 수행기관 조회(performer) | — | `OrganizationOut` |
| `POST /api/reports/issue` | 발급(채번) | `IssueRequest{run_id, model_name, model_version, note?, issuer?}` | `IssuanceOut` |
| `POST /api/reports/{report_no}/reissue` | 재발급(정정) | `ReissueRequest{note}` | `IssuanceOut` |
| `GET /api/reports/{report_no}` | 발급정보 조회(재오픈) | — | `IssuanceOut` |
| ~~`PUT /api/organization`~~ *(설계에만 존재)* | **구현되지 않았고 앞으로도 만들지 않습니다** — 무인증 상태에서 이 엔드포인트 하나로 이미 발급된 모든 성적서의 기관 표기가 소급 변경됩니다(ISSUES.md G-01) | — | — |

### 응답 스키마(pydantic)

```python
class OrganizationOut(BaseModel):
    org_name: str
    department: str | None
    evaluator: str | None
    contact: str | None
    address: str | None

class IssuanceHistoryItem(BaseModel):
    version: str
    issued_at: str       # ISO8601
    note: str | None

class IssuanceOut(BaseModel):
    report_no: str                 # → meta.reportId
    version: str                   # current_version
    issuer: str                    # → signature.issuer
    issued_at: str                 # → signature.signedAt
    organization: OrganizationOut  # → performer
    history: list[IssuanceHistoryItem]   # → signature.history
```

→ **하나의 `IssuanceOut` 으로 프론트의 `meta.reportId` + `performer` + `signature` 를 한 번에 채운다.**

`POST /reports/issue` 는 **멱등 비슷하게** 동작: 같은 `run_id` 로 두 번 호출하면 신규 채번이 아니라 기존 발급본을 반환(중복 채번 방지). 정정은 명시적으로 `reissue` 로만.

## 6. 발급 상태 흐름 (프론트 UX)

```
[평가 완료, 미발급 = 초안(draft)]
   meta.reportId = "(미발급)" / signature 없음
   표지·발급 섹션: "초안 — 미발급" 안내
        │  사용자: 성적서 화면에서 "발급" 클릭
        ▼
[발급됨(issued) v1.0]   POST /api/reports/issue
   meta.reportId = RPT-2026-0001, signature 확정, run 갱신·캐시
        │  사용자: "정정 발급" 클릭(+사유)
        ▼
[재발급(issued) v1.1]   POST /api/reports/{no}/reissue
   같은 번호, version=v1.1, history 2건
```

- 발급 전(초안)에도 평가 결과(KPI/차트/서술)는 **모두 표시**된다. 발급은 번호·서명만 확정한다.
- 발급 버튼·상태 배지는 7단계 성적서 화면 상단 액션에 추가(인쇄 버튼 옆).

## 7. 프론트 연동 변경

| 현재(하드코딩) | 변경 후 |
|---|---|
| `DEFAULT_PERFORMER` 상수 | `GET /api/organization` 조회(또는 `IssuanceOut.organization`) |
| `buildReportId()` | 제거 → 발급 API 의 `report_no` |
| `buildSignature()` | 제거 → 발급 API 의 `issuer/issued_at/history` |
| `mapWorkflowToFinalReport`: meta.reportId/performer/signature 즉석 생성 | **초안 기본값**(미발급 표기)으로 두고, 발급 시 `IssuanceOut` 으로 덮어씀 |
| `addEvaluationRun(reportId=meta.reportId)` (6단계) | run 식별자는 자체 `run-<uuid>` 유지. 성적서 번호는 발급 시 별도 확정(워크스페이스 목록은 "미발급"/발급번호 표시) |

- `FinalReportData` 에 발급 상태 표현: `meta.reportId` 가 없으면 초안. (필요 시 `meta.issued?: boolean` 추가 검토)
- 발급 호출 후 결과를 `useWorkspaceStore` 의 run.reportData 에 병합·영속(재오픈 시 발급정보 유지).

## 8. 파일·모듈 구조 (백엔드)

```
Capstone_Back/
  app/
    core/database.py      # SQLAlchemy engine/SessionLocal/Base, get_db 의존성
    issuance/models.py    # ORM: Organization, Report, Issuance
    issuance/schemas.py   # (추가) OrganizationOut, IssueRequest, IssuanceOut ...
    issuance/service.py   # 채번·발급·재발급 트랜잭션 로직(라우터에서 분리, 단위테스트 용이)
    issuance/router.py    # 발급/조회 API
    issuance/bootstrap.py # 기관 시드(없으면 INSERT): seed_organization / DEFAULT_ORGANIZATION
    main.py               # 기동 시 init_db(create_all) + seed_organization 호출
  data/app.db             # SQLite 파일 (.gitignore)
  tests/test_issuance.py  # 채번 순번/재발급 버전업/멱등 테스트
```

- 의존성 추가: `requirements.txt` 에 `SQLAlchemy>=2.0`.
- DB 파일은 gitignore. 기동 시 `create_all`(마이그레이션 도구 없이 시작; 추후 Alembic 검토 — §11).
- 기관 시드: 기동 시 organization 0행이면 기본값(현 `DEFAULT_PERFORMER`) INSERT.

## 9. 구현 단계 (Phase)

| Phase | 내용 |
|---|---|
| **A. DB 인프라** | `app/core/database.py`, `app/issuance/models.py`(3테이블), `create_all`+기관 시드, requirements 갱신 |
| **B. 발급 서비스+API** | `app/issuance/service.py`(채번/재발급 트랜잭션), `app/issuance/router.py`(issue/reissue/get/organization), `app/issuance/schemas.py` 추가, `app/main.py` 라우터 등록 |
| **C. 백엔드 검증** | `tests/test_issuance.py`: 연도별 순번(0001,0002), 멱등(같은 run 재호출), 재발급 v1.0→v1.1+history, 동시 채번 충돌 |
| **D. 프론트 연동** | org 조회, 발급 버튼/상태 배지, `IssuanceOut`→meta/performer/signature 바인딩, 하드코딩 제거, run 병합·영속 |
| **E. 통합 검증** | 발급→재발급→재오픈 end-to-end, tsc/build, 적대적 검증 워크플로우 |

## 10. 설계 원칙 일관성

- **가짜데이터 0 원칙 유지**: 미발급 상태는 가짜 번호 대신 "초안(미발급)" 으로 표기(기존 차트/latency 의 placeholder 패턴과 동일).
- **graceful degradation**: 발급 API 실패 시 평가 결과·서술은 그대로 표시되고 발급만 막힌다(분리).
- **단일 출처**: 기관/번호/서명은 백엔드 DB 가 권위. 프론트는 표시만.

## 11. 미결 / 리스크

1. **마이그레이션 도구**: 초기엔 `create_all`. 스키마 변경 잦아지면 Alembic 도입(현재는 과함).
2. **인증 부재**: 발급 API 에 인증 없음(현 시스템 전체가 무인증). 발급자(issuer)는 당분간 기관 기본값. 추후 인증 도입 시 실제 사용자명.
3. **버전 표기 규칙**: `v1.x` minor 증가로 시작. 대규모 정정 시 major(v2.0) 정책은 추후.
4. **run_id 신뢰**: run_id 는 프론트 생성(localStorage). 서버는 이를 키로 재발급 판별 — 멱등성은 run_id 유일성에 의존. 동일 run 의 중복 발급은 막지만, 서로 다른 run 이 같은 모델을 평가하면 별도 번호(정상).
5. **PDF 발급 연계**: 향후 PDF 생성 시 발급본(번호·서명 확정)만 PDF 허용하는 정책 검토.

---

## 12. 구현 진행 현황 (Progress Log)

> 규칙: 작업 이행마다 갱신. 상태(✅완료/🔄진행중/🔜예정)와 변경 파일·검증 결과.

| 항목 | 상태 | 비고 |
|---|---|---|
| 설계 확정(본 문서) | ✅ 완료 | DB=SQLite, 채번=발급시점, 기관=단일, 재발급=버전업 |
| Phase A DB 인프라 | ✅ 완료 | `app/core/database.py`(엔진·세션·Base·get_db·`configure_sqlite`), `app/issuance/models.py`(3테이블), `app/main.py` 기동 시 `init_db`+`seed_organization`(`app/issuance/bootstrap.py`), `.gitignore`(충돌 정리 + `data/app.db` 무시), `requirements.txt`+SQLAlchemy>=2.0 |
| Phase B 발급 API | ✅ 완료 | `app/issuance/schemas.py`(+Organization/Issue/Reissue/Issuance 스키마), `app/issuance/service.py`(채번·멱등·재발급·`bump_version`), `app/issuance/router.py`(issue/reissue/get/organization·PUT), 라우터 등록 |
| Phase C 백엔드 테스트 | ✅ 완료 | `tests/test_issuance.py` 18개 통과 (순번·멱등·재발급·연도경계·bump엣지·API·**파일DB FK·2스레드 동시 채번/재발급**) |
| Phase D 프론트 연동 | ✅ 완료(라이브 미검증) | `issuanceApi.ts`(발급 API + KST 포맷터), `useIssuance.ts`(상태·스토어 영속), `ReportLayout`(발급/정정 버튼 + 상태 배지), `Report.tsx`, `SignatureSection`/`ReportCoverSection`/`EvalScopeSection`(초안·"미발급" 표기), `mapWorkflowToFinalReport`(가짜 번호/서명 생성기 제거→초안 기본값). `issued_at`→KST 변환으로 날짜 일원화. tsc 신규 오류 0 |
| Phase E 통합 검증 | 🔄 부분 | 백엔드 end-to-end(발급→재발급→재오픈, TestClient) ✅, 적대적 리뷰 워크플로우(백 5관점·프론트 3관점) ✅, tsc-delta ✅. **미완**: 백+프론트+브라우저 라이브 클릭(현 환경 제약 → `pnpm install` 후 dev 서버로 수동 확인 필요) |

### 12.1 구현 시 설계 대비 결정/강화 (적대적 리뷰 반영)

- **`run_id` UNIQUE 제약 추가**(§3.2는 INDEX만): §11.4 "멱등성은 run_id 유일성에 의존"을 DB 제약으로 강제 → 동시 발급 중복 채번 원천 차단.
- **`UNIQUE(issuance.report_id, version)` 추가**: 동시/재시도 재발급이 같은 버전을 중복 커밋해 이력이 손상되는 것 방지.
- **BEGIN IMMEDIATE 구현**(§4 명시 사항): `configure_sqlite`가 `isolation_level=None`+`BEGIN IMMEDIATE`+`busy_timeout`으로 쓰기 직렬화 → 동시 발급이 교착·HTTP 500 대신 순번 대기. 서비스는 `IntegrityError`/`OperationalError` 모두 재시도.
- **발급 전 기관 존재 선검사**: 기관 미시드 시 FK 위반을 "채번 충돌"로 오인하지 않도록 정확한 오류 반환.
- **`run_id` 빈/공백 문자열 거부**(pydantic 422): 서로 다른 평가가 한 번호로 병합되는 것 차단.
- **PUT /organization 부분 업데이트**(`exclude_unset`): 요청에 없는 필드를 NULL로 덮어쓰지 않음.
- **`issued_at`는 offset 포함 ISO8601로 방출**: naive-UTC로 인한 KST 날짜 하루 어긋남 방지(표시 포맷은 프론트=Phase D 책임).

### 12.2 Phase D 결정/강화 (프론트 적대적 리뷰 반영)

- **초안/발급 상태 모델**: `meta.reportId === ""` = 미발급(초안). `mapWorkflowToFinalReport`가 가짜 `buildReportId`(timestamp 의사난수)·`buildSignature` 생성기를 제거하고 초안 기본값을 둔다.
- **발급은 평가 완료본에만 허용**: `useIssuance.canIssue`를 `isEvaluated` 로 게이트 → 재로드 후 미평가 draft(빈 KPI/차트)에 성적서 번호가 발급되는 것 차단.
- **`issued_at` KST 변환**: 백엔드 offset ISO를 `formatKstDate/DateTime`으로 변환해 표기 → 서명일/발급일시가 KST로 일관(하루 어긋남 해소), `meta.issuedAt`도 백엔드 값으로 일원화.
- **발급 결과 영속**: `run.reportData`/`run.reportId`에 병합 저장(재오픈·인쇄 탭에서 유지). `isEvaluated` 플래그 보존.
- **초안 표기 일원화**: Cover 문서번호/발급일시, EvalScope 시험기간, Signature 섹션 모두 미발급 시 "(미발급)"/"초안—미발급" 표기.
