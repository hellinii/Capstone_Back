"""[B-02] 평가 표본 수는 서버가 알려줘야 한다 — 프론트가 분포 합계로 추측하면 안 된다.

멀티레이블에서 `class_distribution` 은 '레이블 등장 횟수'다(한 샘플이 레이블 3개를 가지면
3번 센다). 그런데 프론트 `buildFactSheet` 이 M21 미선택 시 혼동행렬이 없어서
**분포 합계를 표본 수로 대신 썼다.** 200행 데이터셋이 408건으로 서술된다.

성적서 6절 '총 검증 건수'(200)와 7·8절 서술(408)이 한 문서에서 다른 값으로 인쇄되고,
grounding 화이트리스트는 그 408 을 근거값으로 갖고 있으므로 환각 방어에도 걸리지 않는다.

정본은 `build_evaluation_frame` 이 확정한 프레임의 행 수다(D-01 이 깔아 둔 바닥).
서버가 그것을 `n_samples` 로 실어 보내면 프론트가 추측할 자리가 없어진다.
"""
import io

from fastapi.testclient import TestClient

from app.main import app
from router_cases import CASES, request_json

client = TestClient(app)


def _evaluate(task: str, key: str = "csv"):
    path = CASES[task][key]
    r = client.post(
        "/api/evaluate",
        files={"file": (path.name, io.BytesIO(path.read_bytes()), "text/csv")},
        data={"data": request_json(task)},
    )
    assert r.status_code == 200, r.text[:300]
    return r.json()


def test_multilabel_n_samples_is_rows_not_label_occurrences():
    """[B-02] 200행 멀티레이블에서 n_samples 는 200 이어야 한다 (분포 합계는 408)."""
    body = _evaluate("multilabel")
    dist_total = sum(body["class_distribution"].values())
    assert dist_total > 200, f"이 픽스처는 레이블 등장 횟수가 행 수를 넘어야 한다: {dist_total}"
    assert body["n_samples"] == 200, (
        f"n_samples={body.get('n_samples')} · 분포 합계={dist_total} — 행 수여야 한다"
    )


def test_binary_n_samples_matches_distribution_sum():
    """[B-02] binary 에서는 분포 합계와 n_samples 가 구조적으로 같다.

    `value_counts()` 가 평가 프레임의 행을 그대로 세기 때문이다. 이 등식이 깨지면
    프레임과 분포가 서로 다른 데이터를 보고 있다는 뜻이다.
    """
    body = _evaluate("binary")
    assert body["n_samples"] == sum(body["class_distribution"].values())


def test_multiclass_n_samples_matches_distribution_sum():
    """[B-02] multiclass 도 마찬가지다."""
    body = _evaluate("multiclass")
    assert body["n_samples"] == sum(body["class_distribution"].values())


def test_n_samples_excludes_dropped_rows():
    """[B-02] n_samples 는 '제거 후 실제로 평가된 행 수'다 — 원본 행 수가 아니다.

    저장소 픽스처(csv_errors)는 확률 범위 오류로 400 을 내므로 쓸 수 없다. 결측만
    있는 입력을 직접 만들어 '제거된 행이 n_samples 에서 빠지는가'만 본다.
    """
    import json

    csv = b"id,y_true,y_pred\n1,1,1\n2,0,0\n3,1,\n4,0,1\n5,1,1\n"  # 3번 행 결측
    data = json.dumps({
        "task_type": "binary",
        "column_mappings": [
            {"column": "id", "role": "sample_id"},
            {"column": "y_true", "role": "y_true"},
            {"column": "y_pred", "role": "y_pred"},
        ],
        "selected_metric_ids": ["M1"],
        "metadata": {"positive_class": "1", "negative_class": "0"},
    })
    r = client.post(
        "/api/evaluate",
        files={"file": ("d.csv", io.BytesIO(csv), "text/csv")},
        data={"data": data},
    )
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["dropped_rows"] == 1, body["dropped_rows"]
    assert body["n_samples"] == 4, f"n_samples={body['n_samples']} — 5행 중 1행 제거 후 4여야 한다"
    assert body["n_samples"] + body["dropped_rows"] == 5


def test_narrative_distribution_total_uses_n_samples():
    """[B-02] 서술의 '평가 데이터셋은 총 N건' 도 분포 합계가 아니라 표본 수를 써야 한다.

    n_samples 만 고치고 서술측을 두면 fact_sheet 는 200 인데 폴백 문안은 408 을
    인쇄하는 자기모순이 남는다.
    """
    from app.narrative.derived import compute_derived
    from app.narrative.schemas import DistributionFact, FactSheet

    fs = FactSheet(
        n_samples=200,
        verdict="PASS",
        score=90.0,
        distribution=DistributionFact(
            class_distribution={"sports": 97, "finance": 107, "news": 102, "tech": 102},
        ),
    )
    derived = compute_derived(fs)
    assert derived["distribution"]["total"] == 200, (
        f"total={derived['distribution']['total']} — 레이블 등장 횟수 합(408)이 아니라 표본 수여야 한다"
    )


def test_narrative_distribution_total_falls_back_to_sum():
    """[B-02] n_samples 가 없으면 종전대로 합계를 쓴다 — 구 클라이언트를 깨뜨리지 않는다."""
    from app.narrative.derived import compute_derived
    from app.narrative.schemas import DistributionFact, FactSheet

    fs = FactSheet(
        n_samples=0,
        verdict="PASS",
        score=90.0,
        distribution=DistributionFact(class_distribution={"0": 130, "1": 70}),
    )
    assert compute_derived(fs)["distribution"]["total"] == 200
