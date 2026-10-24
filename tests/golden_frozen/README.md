# tests/golden_frozen — 판 3 고정 정답 (판 4 라운드 S0)

판 4 코드를 넣는 동안 **판 3 동작이 한 글자도 바뀌지 않았는지** 기계가 확인하게 하는 정답 폴더다.
여기 파일은 정적 파일이고, 테스트는 지금 코드로 같은 값을 계산해 **글자 그대로**(`==`) 비교한다.

## 규칙

- **정답을 다시 만드는 것은 경고 신호다. 정답을 고치지 말고 코드를 고친다.**
- 갱신 경로가 없다. `UPDATE_GOLDEN` 같은 환경변수는 이 폴더에 아무 영향이 없고, 파일이 없으면 테스트가 **실패**한다(만들지 않는다).
- 파일 목록과 sha256 은 `MANIFEST.sha256` 에 박혀 있다(`tests/data/edition3_issued` 포함, 숨은 파일 제외). 파일 하나를 바꾸면 `test_frozen_manifest.py` 도 같이 깨진다.
- **시험 코드도 얼린다.** `tests/frozen_s0.py` 와 `tests/test_frozen_*.py`·`tests/test_edition3_frozen.py`·`tests/test_db_columns.py` 는
  **함수 모양(이름·인자·반환 꼴)이 바뀐 것에 맞추는 수정만** 허용한다. 비교 대상을 거르기·건너뛰기(`skip`·`xfail`·조건부 `return`·
  예 목록 줄이기)·느슨한 비교(포함·정규식·공백 무시·일부 키만 비교)로 바꾸는 것은 금지다.
- 정말로 판 3 동작을 바꿔야 하는 결정이 내려졌다면(사람이 확인한 예외), 바뀐 이유를 커밋 메시지에 적고 아래 "다시 만드는 법"으로 뜬 뒤 diff 를 눈으로 확인하고 복사한다. `MANIFEST.sha256` 도 함께 고친다.

## 무엇이 들어 있나

| 파일 | 무엇 | 비교 | 시험 |
|---|---|---|---|
| `narrative_*.json`, `validate_*.json` (12) | `tests/golden` 의 서술·검증 골든 사본 | 원래 테스트와 같은 계산 → 글자 그대로 | `test_frozen_goldens.py` |
| `evaluate_*.base.json` (4) | `tests/golden` 의 평가 골든 사본 | 판 4 에서 키가 늘어나므로 **옛 키·값 ⊆ 지금 `tests/golden/evaluate_*.json`** | `test_frozen_goldens.py` |
| `inputs/fact_sheet_*.json` (4) | `tests/contract` 계약 예시 사본(이진·다중·멀티·옛 main) | 입력 — 계약 폴더가 바뀌어도 정답이 따라가지 않게 | — |
| `inputs/fact_sheet_synth_*.json` (2) | 합성 입력(아래 "합성 입력") — 계약·발급본이 지나지 않는 규칙 문장 갈래용 | 입력 | `test_edition3_frozen.py`(갈래 단정) |
| `edition3/<예>.templates.json` | 규칙 문장 7칸(키 없음 → `generate_sections(None, req)`) | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.job_prompts.json` | 작업 경로가 장마다 실제로 보내는 system·user·응답 스키마 + 칸별 다시 만들기 user | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.fact_view.json` | 사실 시트 보기 글·고객 입력 보기 글·숫자 허용 목록(정렬) | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.legacy_prompts.json` | 옛 경로(`/api/generate-narrative`) 지시문(system·user·스키마)·허용 목록 | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/issued_checks_RPT-2026-000N.json` (6) | 실제 판 3 발급본 AI 칸 글의 칸별 검문 판정(`check_section`·숫자 검문·최종 판정 모순) | 글자 그대로 | `test_edition3_frozen.py` |

`<예>` 는 11개다(`frozen_s0.JOB_CASES`, 옛 경로 지시문은 `legacy_main`·`binary`).
- 계약 예시 `binary`·`multiclass`·`multilabel`(inputs 사본)
- 실제 발급본 요청 6건 `bc_strong`·`bc_weak`·`digits_strong`·`digits_weak`·`yeast_strong`·`yeast_weak`(`tests/data/edition3_issued/requests`)
- 합성 입력 `synth_inconclusive`·`synth_conditional`(inputs)

`frozen_s0.job_case_raw` 는 계약 3종·합성 입력이면 inputs 사본을, 그 밖이면 발급본 요청 원문을 읽는다.
작업 경로 정답은 요청을 `app.narrative.jobs.parse_request` 로 해석한다(작업 경로와 같게).

실제 발급본 자료는 `tests/data/edition3_issued/` 에 있다.
- `requests/*.request.json`: 발급 당시 서술 요청 본문(바이트 그대로).
- `slots/RPT-2026-000N.json`: 발급본 보관 원문(content_json, sha256 = content_hash 확인)에서 옮긴 서술 7칸 글.
- 짝: 0001 bc_strong · 0002 bc_weak · 0003 digits_strong · 0004 digits_weak · 0005 yeast_strong · 0006 yeast_weak.
  파일 이름·작업 유형·모델명·판정·facts 가 요청과 모두 같음을 추출 때 확인했다.

DB 표·열 이름은 정답 파일이 아니라 `tests/test_db_columns.py` 안에 손으로 적었다.

## 합성 입력 (S0 보강)

계약 예시·발급본 6건만으로는 규칙 문장의 몇 갈래가 정답에 없었다(독립 검토). 계약 예시 `inputs/fact_sheet_binary.json` 을
복사해 아래만 고쳤고, 두 파일 모두 요청 스키마(`NarrativeRequest.model_validate`)를 통과한다. 프론트가 만드는 모양
(문제 제목 `"<종류 이름> (<대상>)"`, 판정 불가 근거 `{cause, reason, metrics, judged_count}`, 고정 문구 12 글)을 따랐다.

| 파일 | 고친 것 | 노린 갈래 |
|---|---|---|
| `fact_sheet_synth_inconclusive.json` | 핵심 지표를 F1 Score(M4)·AUROC(M9)로(`core_metrics_source: client`), M9 는 산출 실패라 `metrics` 에서 뺌, M1 Accuracy 를 비핵심 미달로(`is_core: false`), `verdict: INCONCLUSIVE`, `verdict_detail`(failed_noncore M1·unavailable_core M9·`inconclusive_cause: core_not_computed`), 문제 P1 inconclusive·P2 noncore_fail·P3 core_near_target, `f1_threshold_gain`·`error_ratios` 는 null(점수 없음), `dropped_rows: 4`, 고정 문구는 11a·18 만 | 8.2 판정 불가 원인 문장, 지시문의 판정 사유("판정 불가 원인은 …", "목표에 미달한 비핵심 지표는 …"), 8.3 "다만 종합 판정이 '판정 불가'여서 …", 9절 판정 불가 권고, 제외 행 문장 |
| `fact_sheet_synth_conditional.json` | 1,000건 혼동행렬 TN 800·FP 0·FN 20·TP 180 에서 값을 계산(M1 0.98 목표 0.96 핵심 근소 합격, M3 Recall 0.90 목표 0.95 비핵심 미달, M4 F1 0.947 목표 0.90), 판정 대상 구간을 모두 목표와 떨어지게(`ci_contains_target: false`), `verdict: CONDITIONAL_PASS`, 문제 P1 noncore_fail·P2 core_near_target·P3 label_caution, 고정 문구 17·12·11a·18, 지연시간 측정값(`latency.available: true`, 대화형 사용) | 8.2 "판정 대상 지표의 95% 신뢰구간은 모두 목표값과 떨어져 있다."(근소 지표가 있을 때 — `required_items._ci`), 조건부 합격 판정 문장, 정답 라벨 주의, 지연시간·응답 시간 문장 |

조건부 합격·비핵심 미달·불균형 영향·최약 클래스·라벨 주의 문제는 발급본 요청(bc_weak·digits_*·yeast_*)도 덮는다.
"표본 변동에 따라 판정이 달라질 가능성은 낮다" 갈래도 발급본(bc_weak·digits_weak·yeast_strong)이 덮는다.

## 어떻게 떴나 (S0)

- 계산하는 쪽: `tests/frozen_s0.py`(테스트와 생성 스크립트가 같이 쓴다).
- 운영 main(백엔드 `a7d3168`)을 `git archive origin/main` 으로 풀고, 위 헬퍼·입력 사본만 넣어 생성 스크립트를 돌렸다.
  S0 브랜치 코드로 뜬 결과와 `diff -r` 로 같았다(브랜치의 app 차이는 주석·설명 글뿐). 서술·검증 골든 12개는
  `tests/golden` 원본과 바이트까지 같았다.
- S0 보강(발급본 5건·합성 입력 2개의 작업 경로 정답 21개)도 같은 방법(`git archive a7d3168` 위에 지금 tests/ 를 얹음)으로 떴고,
  브랜치 코드로 뜬 결과와 바이트까지 같았다. 이미 있던 정답 32개는 다시 뜬 결과와 바이트까지 같아 그대로 두었다.

## 라운드 끝 확인 (사람이 돌린다)

```bash
# 1) 정답·입력 사본이 S0 끝 뒤로 한 바이트도 바뀌지 않았다(<S0 끝 커밋> = S0 마지막 커밋 해시).
git diff --exit-code <S0 끝 커밋> -- tests/golden_frozen tests/data/edition3_issued
# 2) 시험 코드 diff 를 사람이 읽는다 — 함수 모양 맞춤 외의 변경(거르기·건너뛰기·느슨한 비교)이 없어야 한다.
git diff <S0 끝 커밋> -- tests/frozen_s0.py tests/generate_s0_goldens.py 'tests/test_frozen_*.py' tests/test_edition3_frozen.py tests/test_db_columns.py
```

## 다시 만드는 법 (예외일 때만)

```bash
# 저장소 루트에서. --out 은 tests/ 밖의 빈 폴더여야 한다(스크립트가 저장소 정답을 바로 덮지 않는다).
python tests/generate_s0_goldens.py generate --out /tmp/s0_out
diff -r /tmp/s0_out tests/golden_frozen   # 바뀐 곳을 눈으로 확인한다
# 발급본 칸 글 다시 뽑기(저장소 밖 DB, 읽기 전용):
python tests/generate_s0_goldens.py extract --db <c_view.db> --requests <요청 폴더> --out /tmp/s0_issued
# 복사한 뒤 MANIFEST 다시 쓰기(tests/ 에서):
#   find golden_frozen data/edition3_issued -type f ! -name README.md ! -name MANIFEST.sha256 ! -name '.*' | LC_ALL=C sort | xargs shasum -a 256 > golden_frozen/MANIFEST.sha256
```
