# 성적서 리포트 백엔드 연동 & LLM 서술 생성 설계

> 작성일: 2026-06-23
> 범위: 프론트(`Capstone_Front`) 리포트 ↔ 백엔드(`Capstone_Back`) 연동, LLM 기반 서술(7·8·9절) 자동 생성 모듈 설계
> 상태: **LLM 서술 모듈 Phase 0~4 + P2-10(latency 컬럼 매핑) 완료**. 잔여: P2-11 조직/발급 메타 DB
> 현재 코드 구조는 docs/ARCHITECTURE.md 참조(이 문서는 설계 근거).

---

## 1. 배경 & 문제 정의

현재 프론트의 최종 성적서(`FinalReportData`)는 백엔드와 일부만 연동되어 있고, **상당수 영역이 MOCK(가짜) 데이터로 채워져 그대로 성적서에 노출**된다.

데이터 흐름(현행):

- 리포트 페이지: `src/pages/report/Report.tsx` → 14개 섹션 컴포넌트
- 데이터 훅: `src/hooks/useReportData.ts`
  - `id === "preview"` **그리고** `rawFile`이 있을 때만 `POST /api/evaluate` 호출
  - 그 외 경로(저장된 run·쇼케이스)는 **백엔드 호출 없이 MOCK** 사용
- 매핑: `src/lib/report/mapWorkflowToFinalReport.ts`
  - **73~80행에서 10개 필드를 통째로 `MOCK_FINAL_REPORT`로 채움** ← 가짜 수치가 성적서에 박히는 직접 경로
- 상수: `src/lib/report/reportConstants.ts` / 목 데이터: `src/data/mockReport.ts`

**백엔드가 실제로 채우는 값은 4가지뿐**(`useReportData.ts`):

1. `kpiResults` 값/판정 — `success_metrics[tcId]`
2. `charts.confusionMatrix` — `success_metrics.M21`
3. `kpiResults[].perClass` — `success_metrics.M22`
4. `datasetDiagnosis` 접두 한 줄 + `dataValidation` 중 '제외된 샘플 수'/'누락값' 2개 — `dropped_rows`

→ 나머지(지연시간, ROC/PR 곡선, 7·8·9절 서술, 검증 표 대부분, 데이터 샘플, 종합 판정 등)는 전부 MOCK.

### 1.1 가장 치명적인 사례

- **종합 판정**: `conclusion.verdict = "PASS"`, `score = 94.4`가 `mockReport.ts`에 박혀 있어 **어떤 모델·데이터를 넣어도 항상 "최종 합격 94.4%"** 표시 (KPI는 불합격인데 배너는 합격인 모순 가능)
- **환각 수치**: 7절 정밀 분석이 "오분류 329건/173건/61.8%/Imbalance 1.14" 같은 **가짜 구체 수치**를 LLM 자동생성인 것처럼 노출
- **ROC/PR 곡선 옆 AUROC `0.962`/AUPRC `0.951`**: `ChartSection.tsx`에 리터럴로 박혀 데이터와도 무관
- **"제외된 샘플 수 = 0건"**: `DataValidationSection.tsx:122`에 `<td>0건</td>`로 하드코딩 → 실제 제외 행이 있어도 항상 0

---

## 2. 하드코딩 / MOCK 인벤토리 (요약)

| 영역 | 현재 소스 | 목표 소스 |
|---|---|---|
| `kpiResults` 값/판정 | 백엔드(preview 경로만) / 그 외 MOCK | 백엔드 (전 경로) |
| `charts.confusionMatrix` | 백엔드(preview 경로만) | 백엔드 (전 경로) |
| `kpiResults[].perClass` | 백엔드 M22(preview 경로만) | 백엔드 (전 경로) |
| `dataValidation` (8항목) | MOCK (2항목만 백엔드) | **`/api/validate-data` (이미 완성됨)** |
| `charts.rocCurve` / `prCurve` | MOCK | 백엔드 신규(곡선 좌표) |
| AUROC/AUPRC 라벨 수치 | JSX 리터럴 `0.962`/`0.951` | `success_metrics`(M9/M10) |
| `latency` | MOCK | **컬럼 매핑 기반 산출 (본 문서 §6)** |
| `datasetSamples` | MOCK 10행 | 업로드 파일 실제 상위 N행 |
| `datasetDiagnosis` 본문 | MOCK | `class_distribution` + M23 실수치 |
| `interpretation` (7절) | MOCK (환각 수치) | **LLM 서술 모듈 (§4)** |
| `conclusion` (8절) | MOCK (`PASS`/94.4 고정) | **verdict/score=규칙(§5) + 서술=LLM(§4)** |
| `recommendations` + `recommendationNarrative` (9절) | MOCK | **LLM 서술 모듈 (§4)** |
| `performer.orgName` / `signature` / `reportId` | 상수/`new Date()` | **DB (§7)** |
| `evalEnv.tools` / 표준 문구 / 면책 문구 | 상수 / JSX 리터럴 | 설정값 또는 유지(고정 양식) |

---

## 3. 백엔드 연동 우선순위 TODO

### 🔴 P0 — 즉시 (백엔드 데이터 이미 존재 / 저비용·고임팩트)

1. ✅ **데이터 검증 연동 (완료)** — `dataValidation` MOCK 8항목 제거하고 **이미 완성된** `POST /api/validate-data`(`app/analysis/validation_router.py`) 결과를 리포트로 연결. (`ValidateDataResponse.validation_details` / `execution_summary`)
2. ✅ **"제외된 샘플 수 = 0건" 버그 수정 (완료)** — `DataValidationSection.tsx`의 `<td>0건</td>`를 실측 `validationSummary.excludedRows`로 교체.
3. ⏸ **리포트 데이터 경로 통일 (재평가: 범위 축소)** — 코드 확인 결과 신선한 run은 `rawFile`이 store에 남아 이미 `/api/evaluate`를 호출함. 실제 잔여 문제는 "백엔드가 절대 교체하지 않는 MOCK 필드"(latency·ROC/PR·interpretation·conclusion·recommendations)이며 이는 P1/P2 소관. 쇼케이스(파일 없음) 폴백만 여전히 MOCK base 사용 → P1/P2에서 정리.
4. ⏸ **누락 지표 연동 (재평가: 대부분 이미 동작)** — `useReportData.ts`의 success_metrics 매핑 루프가 선택된 모든 지표를 일반 처리하므로 MCC(M20)/ImbalanceRatio(M23)는 평가 경로에서 이미 반영됨. KPI 그룹 화이트리스트(M1/M4/M20/M23, `KpiResultSection.tsx:10`)는 표시용 고정값으로 현행 유지(추후 사용자 선택 반영 검토).
5. 🔜 **`conclusion.verdict`/`score` 규칙 산출** — 가짜 `PASS`/`94.4` 제거 (§5 규칙). verdict_rules.py(백엔드)에 속하며 §11 "핵심 지표 정의" 확정 필요 → Phase 0에서 진행.

### 🟠 P1 — 백엔드 소규모 추가 필요

6. **ROC/PR 곡선** — `app/evaluation/metrics/binary.py`에 `sklearn.metrics.roc_curve`/`precision_recall_curve` 기반 곡선 좌표 함수 추가. `ChartSection.tsx:21-24`의 `auroc={0.962}`/`auprc={0.951}` 하드코딩 제거 → `success_metrics`(M9/M10).
7. **`datasetSamples`** — 가짜 10행 → 업로드 파일 실제 상위 N행 (프론트가 `rawFile` 보유 → 파싱 가능).
8. **`datasetDiagnosis` 본문** — 가짜 수치 → `metadata.class_distribution` + M23 실수치 기반 재구성.

### 🟡 P2 — LLM 서술 모듈 + 인프라

9. **LLM 서술 3종** (7절 정밀분석 / 8절 종합소견 / 9절 권고안) — §4 설계.
10. **latency** — 컬럼 매핑 기반 산출 — §6 설계.
11. **조직/발급 메타** — `performer.orgName` / `signature` / `reportId` 채번 → **DB** (§7).

---

## 4. LLM 서술 생성 모듈 설계

### 4.1 핵심 철학: LLM은 "수치 생산자"가 아니라 "번역기"

> 모든 숫자는 `app/evaluation/`이 계산한 값에서만 나오고, LLM은 그것을 한국어 산문으로 **옮기기만** 한다.
> 덧셈·반올림·백분율 환산조차 서버가 미리 계산해서 LLM에 먹인다.

이것이 현재 MOCK이 "329건/61.8%/Imbalance 1.14" 같은 **환각 수치를 성적서에 박는 문제**를 원천 차단한다.

### 4.2 아키텍처

```
[POST /api/evaluate] ── 결정적 계산(빠름) ──> success_metrics, M21/22, metadata, latency stats
        │  (프론트 useReportData가 fact_sheet 조립)
        ▼
[POST /api/generate-narrative]   ← 별도 엔드포인트 (확정)
   1. build_fact_sheet — 정규화·반올림·파생합계(오분류수/FP/FN) 서버가 미리 계산
   2. compute_verdict_and_score — 결정론적 규칙 (LLM 아님, §5)
   3. benchmark position — 정적 기준표(app/narrative/baselines.py) 룩업
   4. LLM 호출 — gpt-4.1-nano + strict json_schema + temperature=0 + seed
   5. verify_grounding — 출력 숫자 화이트리스트 검증 ──위반시──> 규칙기반 폴백
```

**평가와 서술을 분리**하는 이유:

- 서술 LLM이 죽어도 "지표는 계산됐는데 평가 전체가 500"이 되지 않게 (독립 try/catch)
- KPI/차트는 즉시 렌더, 서술은 후속 로딩 / 재생성 가능
- `fact_sheet`(수 KB)만 재전송하므로 원시 CSV 재업로드 불필요

### 4.3 4대 불변식 (성적서 신뢰성 보장)

1. **환각 수치 0** — `fact_sheet` 외 데이터 미주입 + 출력 후 숫자 grounding 화이트리스트 검증
2. **판정 무결성** — `verdict`/`score`는 규칙 함수가 단독 산출, LLM은 echo만 받고 불일치 시 서버값으로 덮어쓰기
3. **graceful fallback** — 무키 / API실패 / grounding실패 3경로 모두 규칙기반 폴백. **절대 MOCK으로 회귀 금지**
4. **재현성** — `temperature=0` + `seed` 고정. 같은 입력 → 같은 성적서

### 4.4 신규 파일 (기존 `analyzer.py` 3분할 패턴 답습)

| 파일 | 역할 |
|---|---|
| `app/narrative/router.py` | `POST /api/generate-narrative` 라우터 (`request.app.state.openai_client` 재사용) |
| `app/narrative/service.py` | 오케스트레이션: LLM 호출·폴백 분기·verdict 서버 강제 (grounding/derived 는 별도 모듈) |
| `app/narrative/derived.py` | 파생 계산: 혼동행렬 합계·분포 백분율 등 (`compute_derived`) |
| `app/narrative/grounding.py` | 환각 방어: 숫자 화이트리스트 구축·검증 (`build_number_whitelist`/`verify_grounding`) |
| `app/narrative/prompt.py` | system/user 프롬프트 (환각금지·latency규칙·verdict echo·source_note 의무) |
| `app/narrative/fallback.py` | 규칙기반 템플릿 서술 (`analyzer.analyze_columns_fallback` 동형) |
| `verdict_rules.py` | `compute_verdict_and_score(fact_sheet)` 순수함수 (§5) |
| `app/narrative/baselines.py` | `task_type × metric → {range, source_note}` 정적표 + `get_baseline()` |
| `app/narrative/schemas.py` (추가) | `FactSheet`, `NarrativeRequest`, `NarrativeResponse` pydantic 모델 |

### 4.5 입력 — FactSheet (LLM에 주입할 "사실 시트")

원시 CSV/DataFrame/확률은 **절대 미주입**. 오직 계산된 fact만:

- `task_type`, `n_samples`, `dropped_rows`, `warnings`
- `metrics: [{tc_id, display_name, value(round4·문자열), threshold|null, status: pass/fail/warning}]` — 성공한 선택 지표만. 값은 서버가 문자열로 직렬화해 LLM이 자릿수 변경 못 하게 고정
- `per_class` — M22 classification_report의 클래스별 precision/recall/f1/support
- `confusion: {labels, matrix}` + `derived: {total_misclassified, fp, fn, per_cell}` — **덧셈은 서버가** 미리 계산
- `distribution` — `class_distribution` + `imbalance_ratio`(M23)
- `thresholds_table` — 지표별 기준치 (프론트 `metricDetails`)
- `benchmark_refs: [{metric, model_value, ref_low, ref_high, position, source_note}]` — §4.8
- `verdict_decision: {verdict, score, reasons}` — §5에서 서버가 미리 산출해 주입
- `latency: {available, mean, p50, p95, p99, ...}` — §6 (컬럼 미매핑 시 `{available:false}`)

### 4.6 출력 — NarrativeResponse (structured output)

`response_format = {"type":"json_schema","json_schema":{"name":"narrative_result","strict":true, ...,"additionalProperties":false}}` (analyzer.py 패턴 동일)

```jsonc
{
  "interpretation": {                      // string → 2필드 구조화 (프론트 \n\n split 폐기)
    "confusion_analysis": "string",        // 혼동행렬 기반 오분류/클래스 간섭 해석
    "distribution_analysis": "string"      // 분포·클래스 편향 해석
  },
  "conclusion": {
    "verdict": "PASS|CONDITIONAL_PASS|FAIL",// LLM은 echo만, 서버값으로 덮어씀
    "benchmark": "string",                  // source_note(출처/성격) 포함 의무
    "narrative": "string",
    "risks": "string"
    // score 없음 — 서버 규칙값 사용 (§5)
  },
  "recommendation_narrative": { "data_quality": "string", "model_ops": "string" },
  "recommendations": [
    { "priority": "HIGH|MEDIUM|LOW", "category": "string",
      "action": "string", "expected_impact": "string" }      // maxItems 5
  ],
  "meta": {                                  // 추적성
    "source": "llm|fallback", "model": "string",
    "grounding": { "checked": 0, "violations": [], "passed": true },
    "generated_at": "string"
  }
}
```

> 모델: `gpt-4.1-nano` (analyzer와 동일), `temperature=0`, `top_p=1`, `seed=4213`.
> snake_case(백엔드) → camelCase(프론트) 매핑은 `useReportData`에서 수행.

### 4.7 환각 차단 (grounding 검증) — 핵심

**2중 방어:**

1. **입력측**: `fact_sheet` 외 미주입 + system prompt 강제 규칙
   ("이 JSON 안의 숫자만 사용. 새 숫자 계산·반올림·백분율 환산·추정 절대 금지. 없는 지표·latency 언급 금지.")
2. **출력측** (`grounding.verify_grounding`, 순수 파이썬·테스트 가능):
   - `fact_sheet`의 모든 value를 다중 표기로 전개한 화이트리스트 구축
     예: `0.944` → `{"0.944","0.94","94.4","94"}`, `support=173` → `{"173"}`, matrix 셀·derived 합계·distribution·imbalance·threshold 전부 (정수/소수/×100/반올림 1~2자리 변형 포함)
   - 출력 텍스트 블록에서 정규식 `\d+(?:[.,]\d+)?%?`로 숫자 토큰 추출 → 정규화 후 화이트리스트 대조
   - 화이트리스트 밖 숫자 = 환각 → violation
   - 예외 화이트리스트: 절번호(7,8,9), ISO 연도(2022), 표준번호(4213)
   - **위반 시(엄격 모드, external/project 기본)**: 전체 LLM 산문 폐기 → 규칙 폴백, `meta.grounding.passed=false`
   - pytest로 "환각 숫자 주입 → 폴백 전환" 단위테스트 고정

### 4.8 benchmark 비교 (RAG)

동적 RAG(웹/벡터DB)는 **거부** — 캡스톤 범위 초과 + 재현성 파괴 + 출처불명 환각으로 신뢰성에 정면 위배.

대신 **정적 기준표** `app/narrative/baselines.py`:

- `{task_type × metric → {range_low, range_high, source_note}}` 상수 dict
- 값은 "공개 벤치마크 평균"이라 단정하지 않고 **"내부 참조 기준치"로 라벨링** (허위 권위 방지)
- 서버가 모델 실제 value를 range와 비교해 `position`(above/within/below)을 미리 계산 → `fact_sheet.benchmark_refs`로 주입. LLM은 position을 산문화만 하고 숫자 비교를 직접 안 함
- `conclusion.benchmark` 텍스트에 `source_note` 포함 의무 + "참고용 통상 범위이며 도메인별 상이" 디스클레이머
- 기준표에 없는 metric은 `benchmark_refs`에서 제외 + LLM에 "비교 데이터 없음 → 비교 주장 금지" 지시
- 향후 실 RAG 교체 자리는 `get_baseline()` 1곳에 격리

### 4.9 폴백 (`fallback.build_fallback_narrative`)

발동 3조건(동일 경로): ① `client is None`(무키) ② LLM 호출 예외 ③ grounding 검증 실패

- **절대 `MOCK_FINAL_REPORT` 고정 수치로 회귀 금지** — `fact_sheet` 실수치만 f-string 삽입(환각 0 보장)
- `interpretation`: status별 "{지표} {value}로 기준 {threshold} 대비 {충족/미달}" 문장 (fail 우선 배치) + 오분류 합계·imbalance·dropped_rows 사실 문장
- `conclusion`: verdict/score는 서버 규칙값 그대로, benchmark는 position 템플릿화
- `recommendations`: 정적 매핑 dict (예: Recall 미달 → {HIGH, 클래스 균형, 오버샘플링/가중치}, imbalance>1.5 → 데이터 균형, dropped_rows>0 → 데이터 품질)
- `meta.source="fallback"`, reason(`no_key|api_error|grounding_failed`) 명시
- 폴백도 동일 grounding 검증 통과시켜 회귀 방지 (테스트로 고정)

---

## 5. verdict / score 규칙 (확정)

**판정 대상** = 임계값(목표값)이 설정된 지표만. `"정보 제공"` 지표(Confusion Matrix 등)는 판정 제외.

### verdict

| 조건 | 판정 |
|---|---|
| 대상 지표 **전부 통과** | `PASS` (최종 합격) |
| **일부 미달**이지만 핵심 지표는 통과 | `CONDITIONAL_PASS` (조건부 합격) |
| **핵심 지표 미달** | `FAIL` (최종 불합격) |

### score

```
score = (합격 지표 수 / 전체 대상 지표 수) × 100   // 통과율
```

- 산식은 성적서 부록에 공개 가능

**확정된 세부 규칙 (구현됨)**

- **계산 위치**: 프론트엔드 즉시 계산(`lib/report/computeVerdict.ts`). 임계값(target)이 현재 프론트 `metricDetails`에만 있고 `kpiResults`에 이미 status/threshold가 산출돼 있어, 백엔드 변경 0으로 가짜 PASS를 제거. 추후 백엔드 단일소스가 필요하면 동일 로직을 `verdict_rules.py`로 이관(thresholds를 evaluate 요청에 전송).
- **핵심 지표 (미달 시 FAIL)** = task_type별 코드 상수:
  - binary: Accuracy(M1), F1(M4)
  - multiclass: Accuracy(M1), F1-macro(M4)
  - multilabel: F1(M4), Jaccard(M17)
- **판정 대상** = 임계값(threshold > 0)이 설정된 지표만. "정보 제공" 지표 제외.
- LLM 모듈 도입 시 결과를 `fact_sheet.verdict_decision`으로 주입(서술용), 최종 응답은 항상 규칙값 사용.

---

## 6. Latency 설계 (확정: 컬럼 매핑 방식, 단위 ms 가정)

예측 결과 CSV에 **행별 응답시간 컬럼**이 있으면 컬럼 매핑 단계에서 "응답시간" 역할로 지정 → 백엔드가 그 컬럼을 분석해 통계 산출.

- **컬럼 역할 추가**: `app/core/schemas.py`의 `ColumnRole`에 `latency`(예: `response_time`) 추가 + `VALID_ROLES_BY_TASK` 3개 task 모두 등록
- **백엔드 계산**: 해당 컬럼에서 `mean/min/p50/p95/p99/max` 산출 (`numpy.percentile`) → evaluate 응답 포함. **단위는 ms로 가정.**
- **프론트**: 컬럼 매핑 UI(`src/components/column-mapping/`)에 "응답시간" 역할 추가, `latency` MOCK 제거
- **컬럼 미매핑 시**: 섹션을 "측정 안 됨(N/A)"으로 렌더 + LLM에 미주입(`fact_sheet.latency = {available:false}`)
- LLM은 latency 컬럼이 있을 때만 실제 지연 수치를 인용 (화이트리스트에 포함됨). 없으면 지연 수치 생성 원천 봉쇄

---

## 7. 조직 / 발급 메타 → DB (확정 방향)

다음 값들은 사용자/백엔드 무관하게 상수로 고정되어 있음 → **DB로 분리 예정**:

- `performer` (평가 수행 기관: `orgName` "한국 AI 인증원", `evaluator`, `contact`) — 현재 `reportConstants.ts:DEFAULT_PERFORMER`
- `signature` (발급처 "한국 AI 인증원 평가부", 발급 이력) — 현재 `buildSignature()`
- `reportId` (문서번호) — 현재 `new Date().getTime()` 기반 즉석 생성 → **DB 채번**으로 (재현 가능·추적 가능한 공식 문서번호)
- `evalEnv.tools` (Python/scikit-learn 등 버전) — 설정값 또는 런타임 추출 검토

> 표준 준거 문구(ISO/IEC TS 4213:2022), 면책 4개 항목 등 **고정 양식 문구**는 그대로 유지 가능 (성적서 표준 양식).

---

## 8. 프론트 변경점 요약

- `types/finalReport.types.ts`: `interpretation`을 `string` → `InterpretationData{confusionAnalysis, distributionAnalysis}`. `LatencyStats`를 nullable화. `NarrativeResponse` 타입 추가
- `lib/report/mapWorkflowToFinalReport.ts` (73~80행): **MOCK 주입 제거** → 빈값/플레이스홀더 초기화
- `hooks/useReportData.ts`: evaluate 성공 후 fact_sheet 조립 → `/api/generate-narrative` 호출(독립 try/catch) → snake→camel 매핑 후 4필드 병합
- `components/report/sections/InterpretSection.tsx`: `SUBTITLES` 하드코딩 split 제거 → 2필드 직접 바인딩
- `LatencySection`: 측정치 없으면 N/A 카드 + 디스클레이머
- `DataValidationSection.tsx`: MOCK 8항목 제거 + `/api/validate-data` 연동, "제외 샘플 0건" 버그 수정
- `ChartSection.tsx`: `auroc`/`auprc` 리터럴 제거 → `success_metrics` 바인딩
- `data/mockReport.ts`: 실데이터 경로에서 참조 안 되게 격리, 구체 수치 제거 또는 "예시(가짜)" 표시
- 섹션 컴포넌트: `meta.source === 'fallback'`일 때 "규칙 기반 자동 생성" 배지 (추적성)

---

## 9. 구현 로드맵

| Phase | 내용 |
|---|---|
| **P0 (선행 가능)** | LLM과 독립: `/api/validate-data` 연동, "제외 샘플 0건" 버그, MCC 누락 연동, 데이터 경로 통일 |
| **Phase 0** | 결정론 코어(LLM 무관): `verdict_rules.py`, `app/narrative/baselines.py`, `FactSheet` 스키마, `build_fact_sheet` |
| **Phase 1** | 환각 방어선: `build_fallback_narrative()` + `verify_grounding()` + pytest(환각 주입→폴백 전환 고정) |
| **Phase 2** | LLM 경로: `app/narrative/prompt.py` + `service.generate_narrative()` + `app/narrative/router.py` + `test_narrator.py` |
| **Phase 3** | 프론트 연결: 타입 구조화, MOCK 주입 제거, `useReportData` 조립·병합, 섹션 컴포넌트 수정 |
| **Phase 4** | 통합 검증: 3경로(무키/LLM/환각주입) + 실 CSV e2e(evaluate→narrative→렌더) + FAIL/CONDITIONAL 케이스 verdict 일관성 |

---

## 10. 확정 결정 사항

| 항목 | 결정 |
|---|---|
| LLM 서술 엔드포인트 | 별도 `POST /api/generate-narrative` 신설 |
| verdict 규칙 | 전부 통과=PASS / 일부 미달=CONDITIONAL_PASS / 핵심 미달=FAIL |
| score 규칙 | 통과율 = 합격 지표 수 / 전체 대상 지표 수 × 100 |
| latency | 컬럼 매핑 기반 산출, 단위 **ms 가정** |
| 조직/발급 메타 | **DB로 분리** |
| LLM 모델 | `gpt-4.1-nano`, temp=0, seed 고정 (analyzer.py 패턴 재사용) |
| benchmark | 정적 기준표(내부 참조 기준치 라벨링), 동적 RAG 미채택 |

## 11. 남은 미결 사항 (착수 전 확정)

1. ~~**"핵심 지표" 정의** (§5 verdict 규칙용)~~ → ✅ 확정·구현 완료 (§5 참조: task_type별 상수)
2. DB 스키마 설계 (§7 — performer/signature/reportId 채번)
3. grounding 위반 처리 — 기본 전체 폐기(엄격). internal 한정 부분 교체는 2차 검토

---

## 12. 구현 진행 현황 (Progress Log)

> 규칙: 작업을 이행할 때마다 이 섹션을 갱신한다. 각 항목에 상태(✅완료 / 🔄진행중 / ⏸보류·재평가 / 🔜예정)와 변경 파일·검증 결과를 남긴다.

### 체크리스트

| 항목 | 상태 | 비고 |
|---|---|---|
| P0-1 데이터 검증 실연동 | ✅ 완료 | `/api/validate-data` 결과를 store→매퍼→리포트로 연결 |
| P0-2 제외 샘플 0건 버그 | ✅ 완료 | 실측 `excludedRows` 표시 |
| P0-3 데이터 경로 통일 | ⏸ 재평가 | 신선한 run은 이미 백엔드 호출. 잔여는 P1/P2 MOCK 필드 |
| P0-4 누락 지표(MCC/M23) | ⏸ 재평가 | 평가 경로 일반 루프가 이미 처리. 코드 변경 불요 |
| P0-5 verdict/score 규칙 | ✅ 완료 | 프론트 `computeVerdict.ts`. 가짜 PASS/94.4 + 가짜 서술 제거 |
| P1-6 ROC/PR 곡선 | ✅ 완료 | 백엔드 곡선 좌표(success_metrics.roc_curve/pr_curve) + 프론트 AUROC/AUPRC 하드코딩 제거 |
| P1-7 datasetSamples 실데이터 | 🟡 부분 | 가짜 10행 제거(안내 대체). 풀 구현(실샘플)은 후속 |
| P1-8 datasetDiagnosis 실수치 | ✅ 완료 | class_distribution + M23 imbalance 기반 사실 진단문 |
| 가짜 서술 제거(7·9절) | ✅ 완료 | interpretation/recommendations 플레이스홀더 → 리포트 가짜데이터 0 |
| P2-9 LLM 모듈 Phase 0~2 (백엔드) | ✅ 완료 | 스키마·benchmark·폴백·grounding·프롬프트·`/api/generate-narrative` (폴백 검증) |
| P2-9 LLM 모듈 Phase 3 (프론트 연결) | ✅ 완료 | `buildFactSheet`/`fetchNarrative` 신규, useReportData→generate-narrative 조립·병합, interpretation 구조화, fallback 배지, persist v2 마이그레이션 |
| P2-9 LLM 실호출 검증 | ✅ 완료 | `gpt-4.1-nano` 실호출 end-to-end: source=llm, grounding 통과(위반 0), verdict 서버 강제 확인. 환각 방어선 양방향 실증(미허용 숫자→폴백 차단 / 정확 수치→통과) |
| P2-10 latency 컬럼 매핑 | ✅ 완료 | latency 역할 추가(풀스택), 통계 산출(mean/min/p50/p95/p99/max, ms), 결측·비숫자 best-effort(평가 보존), 미측정 시 placeholder, fact_sheet 반영 |
| P2-11 조직/발급 메타 DB | 🔜 예정 | |

### 작업 로그

#### 2026-06-23 — P0-1 / P0-2 구현 완료

**완료한 것**

- 데이터 검증 결과를 성적서 6절에 실연동. 워크플로우 검증 단계(`/api/validate-data`)에서 받은 결과를 store에 보존하고, 리포트 매핑 시 실제 검증 항목·요약 수치로 변환하도록 변경.
- "제외된 샘플 수" 하드코딩(`0건`) 제거 → 실측값 표시. "총 검증 수행 건수"·"유효 예측 건수"도 실측 요약(`validationSummary`) 기반으로 교체.
- 검증 결과가 없는 경로(쇼케이스 등)에서는 가짜 MOCK 8항목 대신 "검증 결과 없음" 안내를 표시(가짜 데이터 노출 제거).
- `useReportData`에서 `dropped_rows` 기반으로 일부 검증 항목을 덮어쓰던 중복 로직 제거(검증 엔드포인트가 권위 있는 값을 제공하므로).

**변경 파일**

- `types/validation.types.ts` — 백엔드 응답 타입 `ValidateDataResponseData`(snake_case) 단일 진실로 추가
- `types/finalReport.types.ts` — `ValidationSummary` 타입 + `FinalReportData.validationSummary?` 필드 추가
- `lib/report/mapValidationResultToReport.ts` (신규) — 백엔드 검증 응답 → 리포트 `dataValidation`/`summary` 매퍼 (한글 라벨 변환 포함)
- `lib/report/mapWorkflowToFinalReport.ts` — 2번째 인자 `validationResult` 수용, `dataValidation`/`validationSummary` 실데이터 채움(MOCK 제거)
- `components/data-validation/DataValidation.tsx` — 로컬 중복 타입 제거, `validation.types.ts` 재사용
- `utils/stores/useWorkflowStore.ts` — `validationResult` 상태 + setter + reset 처리
- `pages/DataValidation.tsx` — 검증 응답 store 보존 + run 생성 시 매퍼에 전달
- `hooks/useReportData.ts` — 양 경로에서 `validationResult` 전달, 중복 override 제거
- `components/report/sections/DataValidationSection.tsx` — 실측 요약 사용, 빈 상태 안내
- `pages/report/Report.tsx` — `validationSummary` prop 전달

**검증**

- `npx tsc --noEmit`: 변경/신규 파일 0 에러 (기존 무관 에러 5건만 — CSS 모듈 선언, vite.config node 타입, BasicInfo onNext)
- `npx vite build`: 성공 (3210 modules transformed)

**재평가 메모**

- P0-3/P0-4는 코드 정독 결과 문서가 시사한 것보다 범위가 작아 보류(위 표 참조). 실제 고가치 잔여 작업은 P1/P2의 MOCK 필드 교체와 LLM 모듈.

**다음 단계 후보**: P0-5(verdict/score, 단 핵심 지표 정의 선행) 또는 P1-6(ROC/PR 곡선) 또는 LLM Phase 0.

#### 2026-06-24 — P0-5 verdict/score 규칙 산출 완료

**결정 사항** (사용자 확정)

- 계산 위치: **프론트엔드 즉시 계산** (백엔드 변경 0으로 가짜 PASS 제거, 추후 백엔드 이관 가능)
- 핵심 지표: **task_type별 코드 상수** (binary/multiclass: Accuracy(M1)·F1(M4) / multilabel: F1(M4)·Jaccard(M17))

**완료한 것**

- 가짜 "최종 합격 94.4%" 제거 → 실제 KPI 결과 기반 규칙 산출.
  - verdict: 대상 지표(임계값 설정) 전부 통과 → PASS / 일부 미달이나 핵심 통과 → CONDITIONAL_PASS / 핵심 미달 → FAIL
  - score: 합격 지표 수 / 전체 대상 지표 수 × 100 (통과율)
- 종합소견(8절)의 가짜 서술 텍스트(benchmark/narrative/risks의 "94.4%/0.925/0.962" 등) 제거 → LLM 모듈 전까지 "자동 서술 생성 연동 예정" 플레이스홀더 표시. **verdict/score만 실제, 서술은 비워둠**(가짜 데이터 0).

**변경 파일**

- `lib/report/computeVerdict.ts` (신규) — `computeVerdict()`/`buildConclusion()` 순수 규칙 함수
- `lib/report/mapWorkflowToFinalReport.ts` — `conclusion: MOCK` 제거 → `buildConclusion(kpiResults, taskType)`
- `hooks/useReportData.ts` — 평가 경로에서 실측 `updatedKpiResults`로 conclusion 재계산
- `components/report/sections/ConclusionSection.tsx` — 빈 서술 → 플레이스홀더 `NarrativeBlock`

**검증**: `npx tsc --noEmit` 변경 파일 0 에러(기존 무관 5건), `npx vite build` 성공.

**남은 conclusion 작업**: benchmark/narrative/risks 서술 텍스트는 LLM 모듈(P2-9)에서 채움. interpretation(7절)·recommendations(9절)도 여전히 MOCK → P2-9 소관.

**다음 단계 후보**: P1-6(ROC/PR 곡선, 백엔드 곡선 좌표 함수) 또는 P1-7/8(데이터 샘플·진단) 또는 LLM Phase 0(verdict는 이미 프론트에 있으니 benchmark_baselines + FactSheet부터).

#### 2026-06-24 — P1-6 / P1-8 완료, P1-7 부분 처리

**P1-6 ROC/PR 곡선 (완료)**

- 백엔드: `evaluator/metrics/binary.py`에 `calculate_roc_curve`/`calculate_pr_curve` 추가(`sklearn.roc_curve`/`precision_recall_curve`), 60점 균등 다운샘플(`_downsample_pair`). `evaluator/engine.py`에서 binary + AUROC(M9)/AUPRC(M10) 산출 성공 시 `success_metrics.roc_curve`/`pr_curve` 키로 추가(별도 지표·validator 변경 없음).
- 프론트: `charts.rocCurve`에 `auroc?`, `prCurve`에 `auprc?` 추가. `useReportData`가 `success_metrics.roc_curve/pr_curve` + 스칼라 `M9/M10`을 차트에 연결. `ChartSection`의 하드코딩 `auroc={0.962}`/`auprc={0.951}` 제거 → 실제 값 바인딩. 미평가/비binary 경로는 곡선 null + 안내.
- 검증: 백엔드 `engine.evaluate()` 직접 실행으로 곡선 60점·단조성·AUC 스칼라 확인. 기존 pytest 8/9 통과(1건 실패는 **pre-existing** — multilabel M1 미지원, git stash로 확인). 프론트 tsc 0 에러 / vite build 성공.

**P1-8 datasetDiagnosis 실수치 (완료)**

- `lib/report/buildDatasetDiagnosis.ts` 추가 — `metadata.class_distribution`(분포)·imbalance(M23 또는 max/min)·dropped_rows로 사실 기반 진단문 생성. 가짜 MOCK("0.80:0.20, 61.8%, Imbalance 1.14") 제거.
- `mapWorkflowToFinalReport`는 `datasetDiagnosis: ""`로 두고, `useReportData`(미평가/평가 양 경로)가 실데이터로 채움.

**P1-7 datasetSamples (부분)**

- 가짜 10행(MOCK) 제거 → `datasetSamples: []`. `DatasetSection`은 행이 있을 때만 "데이터 예시" 표를 렌더(없으면 미표시).
- **후속 필요(풀 구현)**: 실제 업로드 데이터 상위 N행 표시. 현재 타입이 binary 전용(`{id,y_true,y_pred,score}`)이라 (1) 백엔드가 매핑된 컬럼 기준 샘플 N행 반환 + (2) 타입을 task_type별로 일반화 + (3) DatasetSection 동적 컬럼 렌더가 필요. → P1-7 잔여로 분리.

**변경 파일**: (백엔드) `evaluator/metrics/binary.py`, `evaluator/engine.py` / (프론트) `types/finalReport.types.ts`, `lib/report/mapWorkflowToFinalReport.ts`, `lib/report/buildDatasetDiagnosis.ts`(신규), `hooks/useReportData.ts`, `components/report/sections/ChartSection.tsx`, `components/report/sections/DatasetSection.tsx`

**남은 P1**: P1-7 풀 구현(실샘플). 이후 P2(LLM 모듈, latency 컬럼, DB).

**별건(기존 이슈)**: `test_evaluator.py::test_engine_multilabel_evaluation` 실패 — 백엔드 validator의 multilabel 허용 지표에 M1(Accuracy)이 빠져 있어 발생(프론트 METRICS는 multilabel에 M1 포함). P1과 무관하나 정합성 점검 대상으로 기록.

#### 2026-06-24 — 가짜 서술 제거 + LLM 모듈 Phase 0~2(백엔드) 완료

**1단계: 리포트 가짜 데이터 0 (완료)**

- `mapWorkflowToFinalReport`에서 interpretation(7절)=`""`, recommendationNarrative/recommendations(9절)=빈 값으로 변경(MOCK 제거).
- `InterpretSection`/`RecommendSection`에 빈 값 → "자동 서술 생성(LLM) 연동 예정" 플레이스홀더 추가.
- 결과: 리포트 전 섹션이 **실데이터 또는 플레이스홀더**만 표시(조작 수치/서술 0). tsc 0 에러, vite build 성공.

**백엔드 의존성**: 이 환경 anaconda python에 `fastapi`/`uvicorn`/`openai`/`python-multipart` 설치(pydantic 2.13.4로 업그레이드). 앱 import·기존 테스트 회귀 없음.

**LLM 모듈 Phase 0 (완료)** — `schemas.py`에 `FactSheet`/`NarrativeRequest`/`NarrativeResponse`(+하위 모델). `benchmark_baselines.py` 신규(task_type×지표 내부 참조 기준표 + `get_baseline`/`benchmark_position`/`build_benchmark_refs`, SOURCE_NOTE로 "내부 참조 기준치" 라벨링).

**LLM 모듈 Phase 1 (완료, 환각 방어선)** — `narrator.py`: `compute_derived`(혼동행렬 합계·분포 백분율·판정 카운트 = 서버가 미리 계산), `build_number_whitelist`(다중 표기 화이트리스트 + 절번호/ISO 면제), `verify_grounding`(출력 숫자 추출·대조). `narrative_fallback.py`: 규칙 기반 폴백 서술(fact_sheet 실수치만 삽입). `test_narrator.py` 4/4 통과 — **폴백 자기검증 통과 / 환각 수치(99.7%·41.2ms) 차단 / 면제 토큰 통과** 확인.

**LLM 모듈 Phase 2 (완료, 백엔드 엔드포인트)** — `narrative_prompt.py`(system/user 프롬프트 + strict json_schema), `narrator.generate_narrative`(무키→폴백 / LLM 호출 / grounding 위반→엄격 폴백 / verdict 서버 강제), `routers/narrative.py`(`POST /api/generate-narrative`, app.state 클라이언트 재사용), `main.py` 라우터 등록. TestClient(무키)로 엔드포인트 검증: status 200, source=fallback, 모든 수치 실데이터, 권고 규칙 매핑 정확. 전체 pytest 12 passed / 1 pre-existing fail.

**남음**: LLM 실호출 end-to-end 검증(**`OPENAI_API_KEY` 설정 필요** — 외부 전송·토큰 비용 발생하므로 사용자 승인 후 실행). 키 없어도 폴백으로 동작은 보장됨. 이후 P2-10(latency 컬럼 매핑), P2-11(조직/발급 메타 DB).

#### 2026-06-24 — LLM 모듈 Phase 3 (프론트 연결) 완료

**구현한 것**

- `lib/report/buildFactSheet.ts`(신규) — 평가 결과를 백엔드 `FactSheet`(snake_case)로 조립. KPI→`metrics`(임계값 없으면 `threshold:null, status:"info"`), M21→`confusion`, M22→`per_class`(sklearn `f1-score` 키 처리), metadata→`distribution`, M23→`imbalance_ratio`, verdict/score는 규칙 산출값. n_samples는 혼동행렬 합계>분포 합계 순으로 도출. latency는 `available:false`(P2-10 후속).
- `lib/report/fetchNarrative.ts`(신규) — `POST /api/generate-narrative` 호출 + 응답 snake_case→camelCase 매핑. 호출 실패/비정상 응답 시 빈 서술 반환(**graceful degradation — KPI·차트 렌더는 절대 영향 없음**). priority 값 정규화(HIGH/MEDIUM/LOW). `meta.source`(llm/fallback/error) 전달.
- `hooks/useReportData.ts` — evaluate 후 fact_sheet 조립→`fetchNarrative` 호출→`conclusion={...규칙판정, ...LLM서술}`(verdict/score는 규칙 권위 유지, 백엔드도 강제)·interpretation·recommendationNarrative·recommendations·narrativeSource 병합. await 뒤 `active` 언마운트 가드 추가.
- `types/finalReport.types.ts` — `interpretation: string → InterpretationData {confusionAnalysis, distributionAnalysis}`(백엔드 `InterpretationOut`과 1:1). `NarrativeSource` 타입 + `FinalReportData.narrativeSource?` 추가.
- 섹션 컴포넌트 — `InterpretSection`(구조화 2필드 렌더 + legacy string 방어 정규화), `ConclusionSection`/`RecommendSection`에 `source` prop. 세 섹션 모두 `NarrativeSourceBadge`(신규, 설계 §8: fallback일 때 "규칙 기반 자동 생성" 배지) 표시. `NarrativeBlock`을 `(text ?? "").trim()`로 하드닝(undefined 크래시 방어).
- `data/mockReport.ts`, `lib/report/mapWorkflowToFinalReport.ts` — interpretation을 구조화 형태로 갱신. `pages/report/Report.tsx`·`ReportPrint.tsx`에서 `source={data.narrativeSource}` 전달.
- `utils/stores/useWorkspaceStore.ts` — **persist version 1→2 + `migrate`**: 구버전으로 저장된 `reportData.interpretation`(문자열)을 `{confusionAnalysis, distributionAnalysis}`로 보정(저장된 리포트 rehydration 시 화이트스크린 크래시 방지).

**검증** — 변경분 `tsc --noEmit` 0 에러(기존 무관 5건만 잔존), `vite build` 성공(3215 모듈). 12개 에이전트 적대적 검증 워크플로우(계약 일치/런타임 null 안전성/grounding 정합성/UI 타입/완성도) 실행 → 확정 결함 3건(persist 마이그레이션 누락·문서 미갱신·fallback 배지 미연동) 모두 본 작업에서 수정.

#### 2026-06-24 — LLM 실호출 end-to-end 검증 + grounding 과엄격 개선

**실호출 검증** — `.env` 키로 `TestClient` → `POST /api/generate-narrative` → 실제 `gpt-4.1-nano` 호출(binary, PASS, score=100 케이스).

- **양방향 실증**: (1차) LLM이 화이트리스트 밖 수치 `7.4%`(오분류율)를 출력 → grounding 위반 감지 → external strict mode 폴백 차단 **확인**(`source=fallback, reason=grounding_failed, violations=['7.4%']`). (2차, 개선 후) `source=llm, model=gpt-4.1-nano, grounding.passed=True, violations=[]` — 정상 통과.
- `conclusion.verdict` 가 fact_sheet 값(PASS)으로 서버 강제됨 확인. 7·8·9절 서술 모든 수치가 fact_sheet/derived 실데이터와 일치.

**grounding 과엄격 개선** — `narrator.compute_derived`: 혼동행렬 파생 **백분율**(correct_pct/misclassified_pct, 2x2면 tn/fp/fn/tp_pct)을 서버가 정확히 계산해 derived에 추가. `build_number_whitelist`가 `conf.values()` 전체를 이미 화이트리스트화하므로 자동 반영 → LLM이 오분류율 등 흔한 파생 비율을 인용해도 통과(틀린 비율은 여전히 차단, 안전성 유지). `test_narrator.py` 4/4 통과(회귀 없음).

**남음**: P2-10(아래 항목에서 완료), P2-11(조직/발급 메타 DB). LLM 서술 모듈은 Phase 0~4 전 과정 완료.

#### 2026-06-24 — P2-10 latency 컬럼 매핑 완료 (풀스택)

**구현한 것**

- (백엔드) `schemas.py`: `ColumnRole.latency` 추가 + `VALID_ROLES_BY_TASK` 3개 task_type 모두 허용(선택 역할). `LatencyFact`/`FactSheet.latency`는 기존 정의 재사용. `validator.py`는 무변경(선택 역할이라 필수/지표 요구에 영향 없음).
- (백엔드) `evaluator/preprocessor.py`: latency를 `dropna` 대상에서 제외 → **결측이 분류 평가 샘플을 줄이지 않음**(샘플 보존). latency 컬럼은 `pd.to_numeric(errors="coerce")`로 best-effort 변환(비숫자→NaN, 평가는 계속, 경고). 음수는 경고.
- (백엔드) `evaluator/engine.py`: 메트릭 루프 뒤 `results["latency_stats"]`(count/mean/min/p50/p95/p99/max, unit=ms) 산출(모든 task_type). `report.py`는 무변경(`latency_stats`가 자동으로 success_metrics에 포함).
- (백엔드) `routers/validate.py`: latency 검증 항목(group="latency") 추가 — **결측 제거 후(df_clean) 기준**으로 검사해 평가 결과와 일치. 비숫자/음수는 **error 아닌 warning**(best-effort이므로 평가 진행을 막지 않음).
- (프론트) `data/evaluationData.ts`: `RequiredColumnCode`/`COLUMN_ORDER`/`COLUMN_DISPLAY`에 latency. `ColumnMapping.tsx` roleOptions에 latency.
- (프론트) `lib/mapping/translateRoleToBackend.ts`(신규, 공유 모듈): 평가/검증 양쪽이 쓰던 역할 변환 함수를 단일 출처로 통합(과거 복사본 drift로 검증 경로에 latency 누락 버그가 있었음). `useReportData`·`DataValidation` 모두 import.
- (프론트) `hooks/useReportData.ts`: `success_metrics.latency_stats` → `LatencyStats|null` 추출 → 리포트 `latency` + fact_sheet 반영.
- (프론트) `types/finalReport.types.ts`: `FinalReportData.latency`를 `LatencyStats | null`로(미측정 표현). `LatencySection`: null이면 "미측정" 안내, 값은 `.toFixed(2)`.
- (프론트) `lib/report/mapWorkflowToFinalReport.ts`: 가짜 MOCK latency → `null`(가짜데이터 제거 원칙). `lib/report/buildFactSheet.ts`: `latencyStats`로 `latency.available` 동적 설정. `mapValidationResultToReport.ts`: latency 항목 한글 라벨.

**검증** — `tsc` 0 에러(기존 무관 5건만), `vite build` 성공, 백엔드 pytest 12 passed(사전 존재 multilabel 1건만 실패). latency end-to-end(통계 정확, **결측 행 분류 평가 보존**), validate latency 항목 노출, **비숫자 latency가 평가를 중단시키지 않음**(coerce) 및 **검증이 false blocking을 내지 않음**(warning) 모두 실증.

**적대적 검증 워크플로우(11 에이전트, 5관점)** — 확정 결함 6건(코드 3 + 문서 3) 전부 본 작업에서 수정: (1) 검증 경로 역할 변환에 latency 누락 → 공유 모듈로 추출, (2) validate가 raw df 기준 → df_clean 기준 + 비숫자 warning, (3) 비숫자 latency 평가 abort → coerce, (4~6) 본 문서 §12/헤더/forward-ref 갱신.
