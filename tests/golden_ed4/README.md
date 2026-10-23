# tests/golden_ed4 — 판 4 라운드 정답 (S1~)

판 4 라운드에서 새로 고정한 정적 정답이다. `tests/golden`(`assert_golden` — 파일이 없으면 만들고 통과)과 달리
**갱신 경로가 없다**. 파일이 없으면 시험이 실패한다. 정답을 다시 뜨는 것은 붉은 신호다 — 먼저 코드를 의심한다.

| 파일 | 무엇 | 시험 |
|---|---|---|
| `evaluate_bc_strong.json` | `tests/data/bc_strong`(RPT-2026-0001 원본 CSV + 그때 보낸 평가 요청)의 `/api/evaluate` 응답. `environment` 값은 자리표시자(`golden_utils._redact`) | `test_p4_bc_strong.py` |
| `validate_bc_strong.json` | 같은 입력의 `/api/validate-data` 응답. 운영 main 으로 뜬 프론트 입력 사본(`src/test/edition3/newRun/inputs/bc_strong.validate.json`)과 같았다 | `test_p4_bc_strong.py` |
| `evaluate_bc_strong.edition3_base.json` | 판 3 발급 때(지난 라운드) 실제 평가 응답 원문(바이트 그대로) — 새 응답이 옛 키·값을 모두 그대로 갖는지의 기준 | `test_p4_evaluate_additive.py` |

- 비교: 키 집합·문자열·정수·불리언은 정확히, 실수는 소수 12자리(플랫폼 말단 비트만 흡수).
- 입력 CSV: 3,727바이트, sha256 `4ead9f2ccc8da32c0c3148f02ea6ff4e030c156d9873eeb99883df73f3a49f9d`(시험이 확인).

## 다시 뜨는 법 (예외일 때만)

평가 계산이 바뀌는 결정이 내려졌다면 `ENGINE_VERSION`(app/evaluation/environment.py)을 올리고, 아래로 뜬 뒤 diff 를 눈으로 본다.

```bash
# 저장소 루트에서
PYTHONPATH=.:tests python - <<'EOF'
import json
from fastapi.testclient import TestClient
from app.main import app
from golden_utils import _redact
c = TestClient(app)
csv = open("tests/data/bc_strong/bc_strong.csv", "rb").read()
data = open("tests/data/bc_strong/bc_strong.payload.json").read()
for route, name, redact in (("/api/evaluate", "evaluate", True), ("/api/validate-data", "validate", False)):
    body = c.post(route, files={"file": ("bc_strong.csv", csv, "text/csv")}, data={"data": data}).json()
    body = _redact(body) if redact else body
    open(f"/tmp/{name}_bc_strong.json", "w").write(json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
EOF
diff /tmp/evaluate_bc_strong.json tests/golden_ed4/evaluate_bc_strong.json
```
