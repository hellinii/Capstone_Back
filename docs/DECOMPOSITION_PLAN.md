# 계층화 리팩토링 계획 — 긴 함수/스파게티 세분화

이 문서는 도메인 패키지 재구성(→ [ARCHITECTURE.md](ARCHITECTURE.md)) **이후** 남아 있는, **파일 내부의 긴 함수·뒤엉킨 책임**을 계층적으로 세분화하기 위한 계획이다. 코드 4개 도메인을 병렬 분석해 도출했다.

> 관련: 파일 단위 재배치는 [REFACTORING_PLAN.md](REFACTORING_PLAN.md)에서 완료됨. `schemas.py`의 도메인별 분리는 그 문서의 "PR3"로 별도 관리(여기서 다루지 않음).

---

## 1. 한 줄 진단

> **`issuance` 도메인은 이미 목표 형태**다: 얇은 `router.py`(HTTP만) → `service.py`(오케스트레이션) → 단일책임 헬퍼. **나머지 도메인을 이 패턴으로 끌어올리는 것**이 이번 작업의 전부다.

가장 심각한 스파게티는 **함수당 라인 수**로 드러난다(라인 수 자체보다 이 지표가 중요 — `metrics/*`는 182줄이어도 함수 11개라 건강함):

| 함수 | 라인 | 파일 | 무엇이 뒤엉켰나 |
|---|---:|---|---|
| `validate_data` | **490** | analysis/validation_router.py | 라우터 1개에 HTTP+검증로직 12단계+응답조립 전부 인라인 |
| `preprocess_data` | 150 | evaluation/preprocessor.py | 7~10개 전처리 단계가 한 함수에 |
| `validate_mapping` | 94 | analysis/validator.py | 검증 5단계 순차 인라인 |
| `evaluate` | 93 | evaluation/engine.py | 오케스트레이션 + 부가지표(ROC/PR·latency) 계산 겸함 |
| `extract_metadata` | 92 | analysis/analyzer.py | task_type 3분기 메타계산 + 공통 유니크값 수집 혼재 |
| `issue_report`/`reissue_report` | 68/49 | issuance/service.py | 재시도 루프 스캐폴딩이 복붙·도메인 로직과 엉킴 |
| `generate_narrative` | 64 | narrative/narrator.py | 조율 + LLM호출 세부 + 응답조립 혼재 |
| `build_number_whitelist` 외 | ~57 | narrative/narrator.py | grounding(환각방어) 하위시스템 전체가 오케스트레이터 파일에 상주 |

---

## 2. 목표 계층 & 원칙

```
router.py         HTTP만: 입력 파싱·검증, 예외→상태코드(400/422/500) 매핑, response_model 반환. 도메인 로직 0줄.
   │
service.py        오케스트레이션: 단계 조율, 도메인 예외(XxxError(code,message))로 실패 표현. HTTP를 모름.
   │
헬퍼 모듈들        단일 책임: 한 가지 일만 하는 순수 함수 묶음 (파싱/검증체크/메타계산/grounding 등)
```

적용 규칙:
1. **긴 함수 → 이름 있는 단계 헬퍼**로 추출. 본체는 "헬퍼를 순서대로 부르는 얇은 조율자"로 축소.
2. **`if task_type == ...` 사다리 → 레지스트리 dict**(`{binary: fn, multiclass: fn, ...}`)로 치환.
3. **중복 제거**: 복붙된 재시도 루프·응답 조립·트리거 재계산을 공통 헬퍼로.
4. **도메인 오류는 예외로**(`issuance`의 `IssuanceError` 방식). 상태코드 매핑은 라우터에만.
5. **결정론적 테스트 가능성**: 시각(`now`)·DB 세션·LLM 클라이언트를 파라미터로 주입(이미 `issuance`가 함).

---

## 3. ⚠️ 안전 원칙 (착수 전 반드시)

이번 작업의 **가장 큰 위험은 테스트 공백**이다. 순수 내부 리팩터라 동작이 바뀌면 안 되는데, 아래 3개 라우터는 **HTTP(엔드포인트) 테스트가 아예 없다**:

| 엔드포인트 | 현재 테스트 | 분해 위험 |
|---|---|---|
| `POST /api/validate-data` | ❌ 없음 | **높음** (490줄을 손댐) |
| `POST /api/evaluate` | ❌ 없음 | **높음** |
| `POST /api/generate-narrative` | 함수 단위만(TestClient 없음) | 중간 |
| `POST /api/analyze-columns` | ✅ test_analyze_router | 낮음 |
| `/api/organization`, `/api/reports/*` | ✅ test_issuance | 낮음 |

**따라서 순서는 "테스트 먼저, 리팩터 나중"이다.** 각 대상을 분해하기 전에 **현재 출력을 골든 스냅샷으로 고정하는 characterization 테스트**를 먼저 작성한다. 이후 추출은 "동일 스냅샷 통과 = 동작 불변" 으로 검증된다.

추가 안전망(이미 존재): `tests/test_route_contract.py`(경로 10개 고정) + `pytest`(63개) + CI 부팅 검증.

---

## 4. 우선순위 로드맵 (PR 단위)

작은 PR로 쪼개 각각 독립 배포 가능하게. 위험 낮고 효과 큰 것부터.

| PR | 내용 | 대상 | 위험 | 효과 | 선행 |
|---|---|---|:--:|:--:|---|
| **A** | **안전망**: 3개 무테스트 라우터의 characterization(골든) 테스트 + 추출 예정 순수함수 단위 테스트 | validate-data·evaluate·narrative | 낮음 | 필수 | — |
| **B** | analysis **저위험 순수 추출**: `parsing.py`, `reconcile.py` 분리(동작 불변 이동) | analyzer.py | 낮음 | 중 | — |
| **C** | ★ **validate_data 490줄 분해** → `validation_service.py` + `validation_checks.py`, 라우터 ~30줄로 | validation_router.py | 높음 | **최대** | A |
| **D** | analyzer 나머지 분해 → `llm_mapper.py`·`metadata.py`·`fallback_mapper.py` + `analysis_service.py`(라우터 얇게) | analyzer.py, router.py | 중 | 큼 | A,B |
| **E** | evaluation 계층화 → `service.py` 신설 + `preprocess_data` 단계 헬퍼 + `side_metrics.py` | router·engine·preprocessor | 높음 | 큼 | A |
| **F** | narrative 분해 → `grounding.py`·`derived.py` 분리, `generate_narrative` 얇게 | narrator.py | 중 | 중 | A |
| **G** | issuance 정제(모범 다듬기) → `_commit_with_retry` 중복 제거, `serializers.py`, 시드 → `bootstrap.py` | service·router·(core)database | 낮음 | 중 | — |

> B·G는 선행 없이 지금 바로 가능(저위험). A는 C·D·E·F의 공통 선행. 순서 예: **A → B → C → D → E → F → G** (또는 B·G를 먼저 끼워 워밍업).

---

## 5. 도메인별 상세 분해안

### 5.1 analysis (가장 심각)

**목표 구조**
```
router.py / validation_router.py   (HTTP만)
  → analysis_service.py    resolve_column_mapping(client, task_type, columns, df)  # 무키→룰폴백 / LLM실패→graceful degrade 정책
  → validation_service.py  validate_dataset(df, request) -> ValidateDataResponse   # 검증 파이프라인 조율
  헬퍼:
    parsing.py          parse_file_content + _read_csv_any_encoding + _json_to_df
    llm_mapper.py       analyze_columns_with_llm + _build_response_schema + _call_llm(재시도)
    reconcile.py        reconcile_llm_columns + _norm  (신뢰경계 컬럼 정렬, 순수)
    metadata.py         extract_metadata + task별 빌더 + _detect_binary_classes + 상수
    fallback_mapper.py  analyze_columns_fallback + _guess_role_by_name
    validation_checks.py 개별 점검 함수(공통 6종 + task별 3종 + latency), 각각 -> list[ValidationCheckItem]
    validator.py        (유지) validate_mapping 내부만 _check_*/_compute_tc_availability 헬퍼로
```

**핵심: `validate_data`(490줄)** — 파일파싱→요청파싱→[필수컬럼·결측·중복ID·클래스불일치·제외샘플] 공통 6종 → task_type별(binary/multiclass/multilabel) → latency → 요약, 모두 인라인이고 응답 조립이 2곳(조기반환/정상)에 중복.
→ 라우터는 `parse → validate_dataset(df, req) → 반환`만. `validation_service.validate_dataset`이 파이프라인 조율. task별 분기는 `{binary: check_binary, ...}` 레지스트리로. 각 `check_*`는 `validation_checks.py`의 `(df, mapping_dict, ...) -> list[ValidationCheckItem]` 순수 함수. 응답 중복은 `_build_response(...)`로 통합.
**테스트**: task별 대표 CSV로 현재 `ValidateDataResponse`(항목 name/status/group, error/warning 카운트) 골든 스냅샷 선고정.

**`analyze_columns_with_llm`/`extract_metadata`/`parse_file_content`/`_reconcile_llm_columns`/`analyze_columns_fallback`** — 파싱·LLM I/O·순수변환이 한 파일. 위 헬퍼 모듈로 분리하면 각 규칙을 LLM 없이 단위 테스트 가능. `parsing.py`·`reconcile.py`는 순수 이동이라 **저위험(PR-B로 먼저)**.

### 5.2 evaluation (service 계층 없음)

**목표 구조**
```
router.py    (HTTP만, evaluate_dataset 68→~25줄)
  → service.py (신설)  run_evaluation_pipeline(df, request) : conflict검사→mappings변환→engine.evaluate→전처리error감지→generate_report 조립. 실패는 EvaluationError(code) (issuance 패턴)
  → engine.py  evaluate = 전처리→컨텍스트→디스패치→부가지표부착 5스텝 조율만. 디스패치 루프 _dispatch_metrics로.
  → preprocessor.py  preprocess_data = step 헬퍼 순서 호출로 축소:
       _guard_identical_true_pred_columns / _prune_to_required_columns / _fill_multilabel_missing /
       _drop_missing_rows / _coerce_label_types / _parse_multilabel_columns /
       _validate_probability_columns / _coerce_latency / _check_prob_sum / _extract_class_distribution
  → side_metrics.py (신설)  compute_curve_metrics(ROC/PR), compute_latency_stats  (engine 인라인 계산 이관)
  metrics/*  (유지 — 이미 지표당 함수 1개로 잘 분리됨. 손대지 말 것)
```
**테스트**: `/api/evaluate` characterization(대표 데이터셋의 `EvaluateResponse` 골든) 선고정 — 무테스트 라우터라 위험 높음.

### 5.3 narrative

**목표 구조** (router.py 38줄·baselines.py는 이미 양호 → 유지)
```
narrator.py = service  generate_narrative를 분기 골격만: derived→benchmark_refs→whitelist→_invoke_llm→verify_grounding→_assemble_response/fallback
  헬퍼 추출:
    _invoke_llm(client, req, refs, derived) -> dict   # _MODEL·messages·response_format·temp/seed·json.loads 캡슐화
    _assemble_response(data, fs, grounding) -> NarrativeResponse
  신규 모듈:
    grounding.py  build_number_whitelist(+소스별 _add_* 8종)·verify_grounding·_collect_* + 정규식 상수  (환각 방어, 순수)
    derived.py    compute_derived(+_derive_confusion/_derive_distribution/_derive_counts)·_find_pos_idx
  fallback.py  _recommendations를 _recommendation_signals(1회 계산) + _recommendation_prose + _recommendation_table로(트리거 이원화 중복 제거)
  prompt.py    build_response_schema를 _object 헬퍼 + 섹션별 빌더로(보일러플레이트 감소)
```
`grounding.py`/`derived.py` 분리는 `narrator.py`를 333→얇게 만들고, 환각 방어 규칙을 독립 테스트 가능하게 함. narrator가 재노출(import)해 기존 테스트 호환 유지.

### 5.4 issuance + core (모범 다듬기, 저위험)

이미 최고 수준. 순수 내부 정제만:
- **(A) `_commit_with_retry(db, attempt, *, conflict_code, conflict_message)`** — `issue_report`/`reissue_report`에 **복붙된 재시도 루프** 중복 제거.
- **(B) `_build_new_report(...)` 팩토리 / `_apply_reissue(report, ...)`** — 엔티티 조립·재발급 변이 추출 → 오케스트레이션이 "단계 나열"로.
- **(C) `serializers.py`(신설)** — 라우터의 ORM→Pydantic 직렬화(`_org_out`/`_issuance_out`/`_iso_utc`) 이관. (경미)
- **(D) 계층 역전 정리**: `core/database.py`가 도메인 데이터(`DEFAULT_ORGANIZATION`·`seed_organization`)를 품어 지연 import가 필요함 → **`app/issuance/bootstrap.py`로 이관**하면 정상 import 가능. `main.py`는 `from app.issuance.bootstrap import seed_organization` 한 줄 변경. `init_db`(범용 create_all)는 core 잔류.
- `models.py`(ORM 선언)는 **분해 불필요** — 손대지 말 것.

---

## 6. 손대지 말 것 (과분해 방지)

건강하게 분리돼 있어 건드리면 오히려 나빠지는 것들:
- `app/evaluation/metrics/*` — 지표당 함수 1개, 공유 헬퍼 명확. 그대로.
- `app/narrative/baselines.py`, `app/analysis/prompt_builder.py` — 이미 단일책임.
- `app/issuance/models.py` — ORM 선언 + 제약뿐.
- `core/database.py`의 엔진 설정(`configure_sqlite` 등) — 응집적. 시드 이관(§5.4-D) 외 추가 분해 지양.

---

## 7. 검증 방법 (모든 PR 공통)

1. **착수 전**: 대상 엔드포인트/함수의 **characterization(골든) 테스트** 작성 → 현재 동작 고정 (PR-A).
2. **추출 후**: `pytest -q`(기존 63 + 신규) 전부 통과 = 동작 불변 증명.
3. `tests/test_route_contract.py`로 API 경로 10개 불변 확인.
4. `uvicorn app.main:app` 부팅 + `/health`·대표 엔드포인트 스모크(로컬 + CI 부팅 스텝).
5. 순수 함수로 추출된 헬퍼에는 **테이블 기반 단위 테스트** 신규 추가(엣지 케이스 고정) — 리팩터의 부수 이득.

---

## 8. 요약

- **원칙**: `issuance` 패턴(얇은 라우터 → service → 단일책임 헬퍼)을 전 도메인에 이식. 긴 함수 → 단계 헬퍼, if/elif → 레지스트리, 중복 제거.
- **1순위**: `validate_data`(490줄) 분해 — 단, 무테스트라 **characterization 테스트(PR-A) 선행 필수**.
- **저위험 워밍업**: analysis `parsing.py`/`reconcile.py` 순수 추출(PR-B), issuance 정제(PR-G).
- **불변식**: API 경로·동작 무변경(계약 테스트 + 골든 테스트로 보증), 과분해 금지.
