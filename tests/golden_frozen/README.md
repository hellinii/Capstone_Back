# tests/golden_frozen — 판 3 고정 정답 (판 4 라운드 S0)

판 4 코드를 넣는 동안 **판 3 동작이 한 글자도 바뀌지 않았는지** 기계가 확인하게 하는 정답 폴더다.
여기 파일은 정적 파일이고, 테스트는 지금 코드로 같은 값을 계산해 **글자 그대로**(`==`) 비교한다.

## 규칙

- **정답을 다시 만드는 것은 경고 신호다. 정답을 고치지 말고 코드를 고친다.**
- 갱신 경로가 없다. `UPDATE_GOLDEN` 같은 환경변수는 이 폴더에 아무 영향이 없고, 파일이 없으면 테스트가 **실패**한다(만들지 않는다).
- 파일 목록과 sha256 은 `MANIFEST.sha256` 에 박혀 있다(`tests/data/edition3_issued` 포함, 숨은 파일 제외). 파일 하나를 바꾸면 `test_frozen_manifest.py` 도 같이 깨진다.
- 정말로 판 3 동작을 바꿔야 하는 결정이 내려졌다면(사람이 확인한 예외), 바뀐 이유를 커밋 메시지에 적고 아래 "다시 만드는 법"으로 뜬 뒤 diff 를 눈으로 확인하고 복사한다. `MANIFEST.sha256` 도 함께 고친다.

## 무엇이 들어 있나

| 파일 | 무엇 | 비교 | 시험 |
|---|---|---|---|
| `narrative_*.json`, `validate_*.json` (12) | `tests/golden` 의 서술·검증 골든 사본 | 원래 테스트와 같은 계산 → 글자 그대로 | `test_frozen_goldens.py` |
| `evaluate_*.base.json` (4) | `tests/golden` 의 평가 골든 사본 | 판 4 에서 키가 늘어나므로 **옛 키·값 ⊆ 지금 `tests/golden/evaluate_*.json`** | `test_frozen_goldens.py` |
| `inputs/fact_sheet_*.json` (4) | `tests/contract` 계약 예시 사본(이진·다중·멀티·옛 main) | 입력 — 계약 폴더가 바뀌어도 정답이 따라가지 않게 | — |
| `edition3/<예>.templates.json` | 규칙 문장 7칸(키 없음 → `generate_sections(None, req)`) | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.job_prompts.json` | 작업 경로가 장마다 실제로 보내는 system·user·응답 스키마 + 칸별 다시 만들기 user | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.fact_view.json` | 사실 시트 보기 글·고객 입력 보기 글·숫자 허용 목록(정렬) | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/<예>.legacy_prompts.json` | 옛 경로(`/api/generate-narrative`) 지시문(system·user·스키마)·허용 목록 | 글자 그대로 | `test_edition3_frozen.py` |
| `edition3/issued_checks_RPT-2026-000N.json` (6) | 실제 판 3 발급본 AI 칸 글의 칸별 검문 판정(`check_section`·숫자 검문·최종 판정 모순) | 글자 그대로 | `test_edition3_frozen.py` |

`<예>` 는 계약 예시 `binary`·`multiclass`·`multilabel` 과 실제 요청 `bc_strong` 이다(옛 경로 지시문은 `legacy_main`·`binary`).
작업 경로 정답은 요청을 `app.narrative.jobs.parse_request` 로 해석한다(작업 경로와 같게).

실제 발급본 자료는 `tests/data/edition3_issued/` 에 있다.
- `requests/*.request.json`: 발급 당시 서술 요청 본문(바이트 그대로).
- `slots/RPT-2026-000N.json`: 발급본 보관 원문(content_json, sha256 = content_hash 확인)에서 옮긴 서술 7칸 글.
- 짝: 0001 bc_strong · 0002 bc_weak · 0003 digits_strong · 0004 digits_weak · 0005 yeast_strong · 0006 yeast_weak.
  파일 이름·작업 유형·모델명·판정·facts 가 요청과 모두 같음을 추출 때 확인했다.

DB 표·열 이름은 정답 파일이 아니라 `tests/test_db_columns.py` 안에 손으로 적었다.

## 어떻게 떴나 (S0)

- 계산하는 쪽: `tests/frozen_s0.py`(테스트와 생성 스크립트가 같이 쓴다).
- 운영 main(백엔드 `a7d3168`)을 `git archive origin/main` 으로 풀고, 위 헬퍼·입력 사본만 넣어 생성 스크립트를 돌렸다.
  S0 브랜치 코드로 뜬 결과와 `diff -r` 로 같았다(브랜치의 app 차이는 주석·설명 글뿐). 서술·검증 골든 12개는
  `tests/golden` 원본과 바이트까지 같았다.

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
