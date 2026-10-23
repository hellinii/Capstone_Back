"""
test_narrator.py — LLM 서술 모듈의 환각 방어선(grounding) + 규칙 폴백 단위 테스트.
LLM·API 키 없이 검증 가능한 순수 함수 대상.
"""
import pytest
from pydantic import ValidationError

from app.core.schemas import TaskType
from app.narrative.schemas import ConfusionFact, DistributionFact, FactSheet, MetricFact, NarrativeRequest
from app.narrative.legacy import generate_narrative
from app.narrative.derived import compute_derived
from app.narrative.grounding import (
    build_number_whitelist, verify_grounding, _collect_grounding_texts,
)
from app.narrative.fallback import build_fallback_narrative
from app.narrative.baselines import build_benchmark_refs


def _sample_fact_sheet() -> FactSheet:
    return FactSheet(
        n_samples=200,
        dropped_rows=3,
        verdict="CONDITIONAL_PASS",
        score=66.7,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail"),
            MetricFact(metric_id="M9", display_name="AUROC", value=0.88, threshold=0.80, status="pass"),
        ],
        confusion=ConfusionFact(labels=["0", "1"], matrix=[[120, 10], [15, 55]]),
        distribution=DistributionFact(
            class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.857
        ),
    )


def _all_texts(resp):
    return [
        resp.interpretation.confusion_analysis,
        resp.interpretation.distribution_analysis,
        resp.conclusion.benchmark,
        resp.conclusion.narrative,
        resp.conclusion.risks,
        resp.recommendation_narrative.data_quality,
        resp.recommendation_narrative.model_ops,
        *[r.category for r in resp.recommendations],
        *[r.action for r in resp.recommendations],
        *[r.expected_impact for r in resp.recommendations],
    ]


def test_compute_derived():
    fs = _sample_fact_sheet()
    d = compute_derived(fs)
    assert d["confusion"]["total"] == 200
    assert d["confusion"]["correct"] == 175
    assert d["confusion"]["misclassified"] == 25
    assert d["confusion"]["fn"] == 15 and d["confusion"]["fp"] == 10
    assert d["distribution"]["total"] == 200
    assert d["distribution"]["percentages"]["0"] == 65.0
    assert d["counts"]["target"] == 3 and d["counts"]["passed"] == 2


def test_fallback_passes_own_grounding():
    """폴백 서술의 모든 숫자는 fact_sheet/파생값에서만 나오므로 grounding 을 통과해야 한다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    wl = build_number_whitelist(fs, refs, derived, edition=3)

    resp = build_fallback_narrative(fs, refs, derived, reason="no_key", edition=3)
    grounding = verify_grounding(_all_texts(resp), wl)

    assert grounding.passed, f"폴백이 화이트리스트 밖 숫자 사용: {grounding.violations}"
    assert grounding.checked > 0, "검사된 숫자가 없음(템플릿이 비었는지 확인)"
    assert resp.conclusion.verdict == "CONDITIONAL_PASS"
    assert resp.meta.source == "fallback"


def test_grounding_catches_hallucination():
    """fact_sheet 에 없는 숫자(환각)를 주입하면 검증이 위반으로 잡아야 한다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    wl = build_number_whitelist(fs, refs, derived, edition=3)

    hallucinated = ["정확도는 무려 99.7%에 달하며 P99 지연시간은 41.2ms이다."]
    grounding = verify_grounding(hallucinated, wl)

    assert not grounding.passed
    assert any("99.7" in v for v in grounding.violations)
    assert any("41.2" in v for v in grounding.violations)


def test_exempt_tokens_not_flagged():
    """절 번호/ISO 연도 같은 면제 토큰은 위반으로 잡지 않는다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    wl = build_number_whitelist(fs, [], derived, edition=3)
    g = verify_grounding(["본 시험은 ISO/IEC TS 4213:2022 기준으로 7절·8절·9절을 구성한다."], wl)
    assert g.passed, f"면제 토큰 오탐: {g.violations}"


# ── D2: compute_derived 의 FN/FP 가 positive 클래스 index 를 따라 매핑되는지 ──

def _confusion_fs(labels, matrix, positive_class=None) -> FactSheet:
    return FactSheet(
        verdict="PASS", score=100.0,
        confusion=ConfusionFact(labels=labels, matrix=matrix, positive_class=positive_class),
    )


def test_compute_derived_positive_index0():
    """positive('fraud')가 정렬상 index 0 → 하드코딩(index1)이면 FN/FP 가 반전된다."""
    fs = _confusion_fs(["fraud", "normal"], [[40, 10], [5, 45]], positive_class="fraud")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 40 and c["fn"] == 10 and c["fp"] == 5 and c["tn"] == 45


def test_compute_derived_positive_index1_default():
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class="1")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 55 and c["fn"] == 15 and c["fp"] == 10 and c["tn"] == 120


def test_compute_derived_positive_none_fallback():
    """positive 미지정 → index 1 폴백(기존 동작 하위호환)."""
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class=None)
    c = compute_derived(fs)["confusion"]
    assert c["fn"] == 15 and c["fp"] == 10


def test_compute_derived_positive_not_in_labels_fallback():
    fs = _confusion_fs(["0", "1"], [[120, 10], [15, 55]], positive_class="xyz")
    c = compute_derived(fs)["confusion"]
    assert c["fn"] == 15 and c["fp"] == 10


def test_compute_derived_positive_float_match():
    """positive_class='0.0' 는 라벨 '0'(index 0)과 float 매칭되어야 한다."""
    fs = _confusion_fs(["0", "1"], [[40, 10], [5, 45]], positive_class="0.0")
    c = compute_derived(fs)["confusion"]
    assert c["tp"] == 40 and c["fn"] == 10 and c["fp"] == 5 and c["tn"] == 45


# ── D3a/D3b: grounding 면제 제거 + 커버리지 ──────────────────────────────────

def _no_special_fact_sheet() -> FactSheet:
    """0/1/100/2022 가 화이트리스트에 들어가지 않도록 구성(면제 제거 검증용)."""
    return FactSheet(
        n_samples=200, dropped_rows=3, verdict="PASS", score=72.0,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.88, threshold=0.80, status="pass"),
            MetricFact(metric_id="M4", display_name="F1 Score", value=0.90, threshold=0.80, status="pass"),
        ],
        confusion=ConfusionFact(labels=["A", "B"], matrix=[[130, 12], [11, 47]], positive_class="B"),
        distribution=DistributionFact(class_distribution={"A": 130, "B": 70}, imbalance_ratio=1.857),
    )


def test_bare_special_numbers_flagged():
    """근거 없는 0/1/100/2022 는 더 이상 면제되지 않고 위반으로 잡힌다(D3a)."""
    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    g = verify_grounding(["정확도는 100%이며 제외 0건, 신뢰도 1.0, 총 2,022건이다."], wl)
    assert not g.passed
    assert "100%" in g.violations and "0" in g.violations and "2,022" in g.violations


def test_genuine_perfect_and_zero_pass():
    """실측 100%(=1.0)·0건이 fact_sheet 근거이면 통과한다(과차단 방지)."""
    fs = FactSheet(
        n_samples=100, dropped_rows=0, verdict="PASS", score=100.0,
        metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=1.0, threshold=0.9, status="pass")],
    )
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    g = verify_grounding(["정확도 100%, 제외 0건, 점수 1.0"], wl)
    assert g.passed, g.violations


def test_metric_name_and_ordinal_not_flagged():
    """F1·P99 식별자와 제N종·N절 문맥 토큰은 오탐하지 않는다(D3a)."""
    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    g = verify_grounding(["F1 Score와 P99 지연, 제1종·제2종 오류, 7절 참조."], wl)
    assert g.passed, g.violations


def test_grounding_covers_recommendation_category():
    """recommendations[].category 의 환각 숫자도 전수 수집으로 검증된다(D3b)."""
    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    data = {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [
            {"priority": "HIGH", "category": "재현율 0.777 보강", "action": "", "expected_impact": ""}
        ],
    }
    g = verify_grounding(_collect_grounding_texts(data), wl)
    assert not g.passed
    assert any("0.777" in v for v in g.violations)


# ── D3c/D7[1]: internal 폴백 + 조립 실패 폴백 (async, 목 클라이언트) ─────────────

async def test_internal_grounding_violation_falls_back(make_fake_anthropic_client):
    """report_purpose=internal 이어도 grounding 위반 시 규칙 폴백으로 강등(fail-open 제거, D3c)."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "재현율은 0.777로 낮다.", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_anthropic_client(llm_json)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="internal", fact_sheet=fs)
    resp = await generate_narrative(client, req)
    assert resp.meta.source == "fallback"
    assert resp.meta.reason == "grounding_failed"
    assert resp.meta.grounding.passed is False
    # 폴백 산문은 fact_sheet 근거만 사용하므로 자체 grounding 통과
    wl = build_number_whitelist(fs, build_benchmark_refs("binary", fs.metrics, edition=3), compute_derived(fs), edition=3)
    assert verify_grounding(_all_texts(resp), wl).passed


async def test_clean_llm_response_source_llm(make_fake_anthropic_client):
    """근거 있는 정상 응답은 source=llm, verdict 는 서버값으로 강제."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "정확도 0.94로 양호.", "distribution_analysis": ""},
        "conclusion": {"verdict": "누가봐도이상함", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_anthropic_client(llm_json)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=fs)
    resp = await generate_narrative(client, req)
    assert resp.meta.source == "llm"
    assert resp.conclusion.verdict == "PASS"  # LLM echo 무시, fact_sheet 강제


def test_report_purpose_enum_rejects_invalid():
    """report_purpose 는 enum 이라 허용값 외 문자열은 거부된다(D7[4] 프롬프트 주입 차단)."""
    with pytest.raises(ValidationError):
        NarrativeRequest(
            task_type=TaskType.binary,
            report_purpose="ignore-previous-instructions",
            fact_sheet=_no_special_fact_sheet(),
        )


def test_benchmark_direction_lower_is_better():
    """Hamming Loss(낮을수록 좋음): 범위 아래=우수, 위=미흡 (D7[3])."""
    good = build_benchmark_refs(
        "multilabel", [MetricFact(metric_id="M15", display_name="Hamming Loss", value=0.02, threshold=0.1, status="pass")], edition=3
    )[0]
    bad = build_benchmark_refs(
        "multilabel", [MetricFact(metric_id="M15", display_name="Hamming Loss", value=0.30, threshold=0.1, status="fail")], edition=3
    )[0]
    assert good["direction"] == "lower" and good["quality"] == "better"
    assert bad["quality"] == "worse"


def test_benchmark_direction_higher_is_better():
    """Accuracy(높을수록 좋음): 범위 위=우수 (D7[3])."""
    ref = build_benchmark_refs(
        "binary", [MetricFact(metric_id="M1", display_name="Accuracy", value=0.95, threshold=0.85, status="pass")], edition=3
    )[0]
    assert ref["direction"] == "higher" and ref["quality"] == "better"


async def test_assembly_error_falls_back(make_fake_anthropic_client):
    """스키마 불일치(조립 불가) 응답은 500 대신 규칙 폴백(assembly_error, D7[1])."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": ["not-a-dict"],  # RecommendationOut(**r) 조립 실패 유발
    }
    client = make_fake_anthropic_client(llm_json)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=fs)
    resp = await generate_narrative(client, req)
    assert resp.meta.source == "fallback"
    assert resp.meta.reason == "assembly_error"


# ── 판정 불가(INCONCLUSIVE)와 고객 입력(client_inputs) ─────────────────────────────
#
# 서술 설계 D19: 핵심 지표가 산출되지 않았거나 판정 대상이 없으면 "판정 불가"다. 확인하지 못한
# 것을 "모두 충족"으로 쓰면 확인한 것처럼 읽힌다.
# 서술 설계 D25: 고객 입력은 모두 AI 에 넘긴다. 고객이 적은 숫자를 서술이 옮겨 적는 것은 환각이
# 아니므로 숫자 허용 목록에 들어가야 한다.

def test_inconclusive_rejects_all_passed_claim():
    from app.narrative.grounding import find_verdict_contradictions

    texts = ["모든 시험항목이 합격 기준을 충족하였다."]
    assert find_verdict_contradictions(texts, "INCONCLUSIVE") == texts
    # 판정 불가에서 일부 지표의 미달을 적는 것은 사실일 수 있어 막지 않는다.
    assert find_verdict_contradictions(["Recall 이 목표에 미달하였다."], "INCONCLUSIVE") == []


def test_fallback_labels_inconclusive():
    fs = _sample_fact_sheet().model_copy(update={"verdict": "INCONCLUSIVE"})
    benchmark_refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    resp = build_fallback_narrative(fs, benchmark_refs, compute_derived(fs), reason="no_key", edition=3)
    assert resp.conclusion.verdict == "INCONCLUSIVE"
    assert "판정 불가" in resp.conclusion.narrative


def test_client_input_numbers_are_allowed():
    fs = _sample_fact_sheet()
    benchmark_refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    derived = compute_derived(fs)
    client_inputs = {
        "training_data": {"학습 샘플 수": "12,000개", "수집 기간": "2023-01 ~ 2024-12"},
        "ground_truth": {"라벨러 수": "3명", "라벨 일치도": "Cohen의 카파 0.71"},
    }
    text = ["학습 데이터 12,000개, 라벨러 3명, 카파 0.71 로 보고되었다."]

    without = verify_grounding(text, build_number_whitelist(fs, benchmark_refs, derived, edition=3))
    assert not without.passed

    with_inputs = build_number_whitelist(fs, benchmark_refs, derived, client_inputs, edition=3)
    assert verify_grounding(text, with_inputs).passed


def test_client_inputs_size_limited():
    from app.narrative.schemas import MAX_CLIENT_INPUTS_CHARS

    fs = _sample_fact_sheet()
    ok = NarrativeRequest(task_type=TaskType.binary, fact_sheet=fs, client_inputs={"a": "b"})
    assert ok.client_inputs == {"a": "b"}
    with pytest.raises(ValidationError):
        NarrativeRequest(
            task_type=TaskType.binary,
            fact_sheet=fs,
            client_inputs={"note": "x" * (MAX_CLIENT_INPUTS_CHARS + 1)},
        )


def test_client_inputs_length_is_compact_json_like_the_frontend():
    """B7: 공백 없는 JSON 으로 잰다 — 프론트 JSON.stringify 와 같은 수. 한글은 1자로 센다."""
    from app.narrative.schemas import client_inputs_length

    v = {"declarations": {"사용 방식": "배치", "목록": [1, 2]}, "note": "가나다"}
    expected = '{"declarations":{"사용 방식":"배치","목록":[1,2]},"note":"가나다"}'
    assert client_inputs_length(v) == len(expected)


def test_client_inputs_limit_boundary_is_exact():
    """B7: 직렬화 길이가 정확히 상한이면 통과, 한 자 더 많으면 422(ValidationError)."""
    from app.narrative.schemas import MAX_CLIENT_INPUTS_CHARS, client_inputs_length

    fs = _sample_fact_sheet()
    # {"a":{"b":"…","c":"…"}} — 기본 구분자였다면 공백 3자가 더해져 상한을 넘는 꼴로 만든다.
    base = {"a": {"b": "", "c": "가"}}
    filler = MAX_CLIENT_INPUTS_CHARS - client_inputs_length(base)
    at_limit = {"a": {"b": "나" * filler, "c": "가"}}
    assert client_inputs_length(at_limit) == MAX_CLIENT_INPUTS_CHARS
    ok = NarrativeRequest(task_type=TaskType.binary, fact_sheet=fs, client_inputs=at_limit)
    assert ok.client_inputs == at_limit

    over = {"a": {"b": "나" * (filler + 1), "c": "가"}}
    with pytest.raises(ValidationError):
        NarrativeRequest(task_type=TaskType.binary, fact_sheet=fs, client_inputs=over)


def test_client_inputs_reach_the_prompt():
    from app.narrative.legacy import build_user_prompt

    prompt = build_user_prompt({"verdict": "PASS"}, [], {}, {"declarations": {"사용 방식": "배치"}})
    assert '"client_inputs"' in prompt and "배치" in prompt
    # 없으면 싣지 않는다 — 종전 요청과 같은 프롬프트다.
    assert "client_inputs" not in build_user_prompt({"verdict": "PASS"}, [], {})


def test_metric_fact_core_flag_defaults_false():
    assert MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, status="pass").is_core is False


def test_class_average_rows_reach_the_fact_sheet_and_whitelist():
    """클래스별 표(M22)의 평균 줄을 버리지 않는다(서술 설계 D17·D39). 서술이 그 숫자를 쓰면 허용된다."""
    from app.narrative.schemas import PerClassFact

    fs = _sample_fact_sheet().model_copy(
        update={
            "class_averages": [
                PerClassFact(label="매크로 평균", precision=0.731, recall=0.682, f1=0.705, support=200),
                PerClassFact(label="가중 평균", precision=0.794, recall=0.781, f1=0.786, support=200),
            ]
        }
    )
    derived = compute_derived(fs)
    wl = build_number_whitelist(fs, [], derived, edition=3)

    g = verify_grounding(["매크로 평균 F1 은 0.705, 가중 평균 F1 은 0.786 이다."], wl)
    assert g.passed, g.violations


def test_class_averages_are_bounded():
    """평균 줄은 많아야 넷이다(매크로·가중·마이크로·샘플). 그보다 많으면 거절한다."""
    from app.narrative.schemas import PerClassFact

    rows = [PerClassFact(label=f"avg{i}", f1=0.5) for i in range(5)]
    with pytest.raises(ValidationError):
        FactSheet(verdict="PASS", score=100.0, class_averages=rows)


def test_system_prompt_names_the_task_type():
    """분류 유형 이름이 AI 에 직접 간다(재료 목록 2절 "작업 유형 — AI 예")."""
    from app.narrative.legacy import build_system_prompt

    assert "이진 분류" in build_system_prompt("internal", "binary")
    assert "다중 클래스 분류" in build_system_prompt("internal", "multiclass")
    # W9: 한국어 이름은 '다중 레이블 분류'로 통일한다(옛 '멀티레이블 분류'는 AI 에 가지 않는다).
    multilabel = build_system_prompt("internal", "multilabel")
    assert "다중 레이블 분류" in multilabel
    assert "멀티레이블 분류" not in multilabel


def test_system_prompt_core_metric_line_is_neutral():
    """W8: 핵심 지표는 의뢰자가 바꾸지 않으면 분류 유형의 기본값이다. AI 지시문이 '의뢰자가 지정'이라고
    단정하면 성적서 6.1(기본값 사용)과 서술이 갈린다. 세 유형 모두 중립 문장만 간다."""
    from app.narrative.legacy import build_system_prompt

    for task_type in ("binary", "multiclass", "multilabel"):
        prompt = build_system_prompt("internal", task_type, _sample_fact_sheet())
        assert "is_core=true 는 판정에 쓰는 핵심 지표다(의뢰자가 정했거나 분류 유형의 기본값)" in prompt
        assert "의뢰자가 지정" not in prompt


def test_task_type_labels_are_korean_names():
    """W9: 프론트 성적서 TASK_TYPE_LABELS 와 같은 세 이름(프론트 대조 테스트가 이 표를 읽는다)."""
    from app.narrative.prompt import TASK_TYPE_LABELS

    assert TASK_TYPE_LABELS == {
        "binary": "이진 분류", "multiclass": "다중 클래스 분류", "multilabel": "다중 레이블 분류",
    }


def test_derived_prediction_threshold_is_allowed_in_narrative():
    """파생 예측 사실이 사실 시트에 실려 가고(재료 목록 9절), 그 임계값을 서술에 써도 된다."""
    from app.narrative.schemas import DerivedPredictionFact

    fs = _sample_fact_sheet().model_copy(
        update={
            "derived_prediction": DerivedPredictionFact(
                method="threshold", threshold=0.35, source_columns=["score"]
            )
        }
    )
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)

    assert verify_grounding(["예측값은 확률에 임계값 0.35 를 적용해 만들었다."], wl).passed


def test_per_label_thresholds_are_allowed_in_narrative():
    from app.narrative.schemas import DerivedPredictionFact

    fs = _sample_fact_sheet().model_copy(
        update={
            "derived_prediction": DerivedPredictionFact(
                method="threshold_per_label", threshold={"score_a": 0.4, "score_b": 0.65}
            )
        }
    )
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)

    assert verify_grounding(["레이블 a 는 0.4, b 는 0.65 로 판정했다."], wl).passed


async def test_benchmark_refs_come_back_with_the_narrative():
    """8.1 에 참조 범위를 숫자로 찍도록 비교값을 응답에 싣는다(재료 목록 9절). 폴백이어도 같다."""
    req = NarrativeRequest(task_type=TaskType.binary, fact_sheet=_sample_fact_sheet())
    resp = await generate_narrative(None, req)

    expected = build_benchmark_refs("binary", req.fact_sheet.metrics, edition=3)
    assert resp.meta.source == "fallback"
    assert [r.model_dump() for r in resp.benchmark_refs] == expected
    assert len(expected) > 0


def test_confusion_counts_reach_the_fact_sheet_and_whitelist():
    """TP·TN·FP·FN 건수는 M21 없이도 AI 에 간다(재료 목록 9절). 서술이 그 건수를 쓰면 허용된다."""
    from app.narrative.schemas import ConfusionCountFact

    fs = _sample_fact_sheet().model_copy(
        update={
            "confusion": None,
            "confusion_counts": [ConfusionCountFact(label="1", tp=52, fp=56, fn=43, tn=49)],
        }
    )
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)

    assert verify_grounding(["양성 기준 TP 52건, FN 43건이다."], wl).passed


@pytest.mark.parametrize("target", [0.0, -0.1])
def test_zero_or_negative_target_is_a_target(target):
    """목표값 0·음수(MCC)도 목표다 — 미달이면 폴백이 '모두 충족'이라 쓰면 판정과 모순된다."""
    fs = FactSheet(
        verdict="FAIL",
        score=0.0,
        metrics=[
            MetricFact(metric_id="M20", display_name="MCC", value=-0.2, threshold=target,
                       status="fail", is_core=True),
        ],
    )
    d = compute_derived(fs)
    assert d["counts"]["target"] == 1
    assert d["counts"]["passed"] == 0
    resp = build_fallback_narrative(fs, [], d, "no_key", edition=3)
    assert "MCC" in resp.conclusion.risks
    assert "모든 시험항목이 기준을 충족" not in resp.conclusion.risks
    assert "1개 중 0개" in resp.conclusion.narrative


def _with_derivation(method, threshold=None):
    from app.narrative.schemas import DerivedPredictionFact

    return _sample_fact_sheet().model_copy(
        update={"derived_prediction": DerivedPredictionFact(method=method, threshold=threshold)}
    )


def test_rule_10_follows_the_derivation_method():
    """규칙 10 은 파생 방법마다 다르다 — 다중 클래스(argmax)에는 임계값이 없다(DESIGN_GAPS R3)."""
    from app.narrative.legacy import build_system_prompt

    threshold = build_system_prompt("internal", "binary", _with_derivation("threshold", 0.35))
    per_label = build_system_prompt("internal", "multilabel", _with_derivation("threshold_per_label", {"a": 0.4}))
    argmax = build_system_prompt("internal", "multiclass", _with_derivation("argmax"))
    direct = build_system_prompt("internal", "binary", _sample_fact_sheet())

    assert "10. " in threshold and "결정 임계값을 적용" in threshold
    assert "10. " in per_label and "레이블마다" in per_label
    assert "10. " in argmax and "가장 큰 클래스" in argmax
    assert "임계값을 적용" not in argmax
    # 예측을 직접 올렸으면 파생 사실이 없으니 규칙 10 도 없다.
    assert "10. " not in direct and "derived_prediction" not in direct


async def test_service_sends_the_method_specific_rule(make_fake_anthropic_client):
    llm_json = {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_anthropic_client(llm_json)
    req = NarrativeRequest(
        task_type=TaskType.multiclass, report_purpose="internal", fact_sheet=_with_derivation("argmax")
    )
    await generate_narrative(client, req)

    system = client.beta.messages.create.call_args.kwargs["system"]
    assert "가장 큰 클래스" in system
    assert "임계값을 적용" not in system


# ── 목표값 수준(서술 설계 D31) ─────────────────────────────────────────────────
#
# 참조 범위와 목표값이 만나는 곳: 목표를 범위보다 낮게/안에/높게 잡았는지를 단순 수치 위치로
# 싣는다(방향 반영 없음). 목표값 0·음수도 실제 목표다 — None 일 때만 '목표 없음'.

def _target_ref(threshold, value=0.88, name="Accuracy", task="binary"):
    m = MetricFact(metric_id="M1", display_name=name, value=value, threshold=threshold, status="info")
    return build_benchmark_refs(task, [m], edition=3)[0]


@pytest.mark.parametrize("threshold, expected", [
    (0.84, "below"),
    (0.85, "within"),   # ref_low 와 같으면 범위 안
    (0.9, "within"),
    (0.92, "within"),   # ref_high 와 같으면 범위 안
    (0.93, "above"),
])
def test_target_position_boundaries(threshold, expected):
    ref = _target_ref(threshold)  # binary Accuracy 참조 범위 0.85~0.92
    assert ref["target_value"] == threshold
    assert ref["target_position"] == expected


@pytest.mark.parametrize("threshold", [0.0, -0.1])
def test_target_zero_or_negative_is_a_real_target(threshold):
    ref = _target_ref(threshold)
    assert ref["target_value"] == threshold
    assert ref["target_position"] == "below"


def test_no_target_gives_null_target_fields():
    ref = _target_ref(None)
    assert ref["target_value"] is None and ref["target_position"] is None


def test_target_position_ignores_direction():
    """낮을수록 좋은 지표(Hamming Loss)도 목표값 위치는 단순 수치 위치다(우수/미흡 아님)."""
    ref = _target_ref(0.02, value=0.1, name="Hamming Loss", task="multilabel")  # 범위 0.05~0.20
    assert ref["target_position"] == "below"


def test_fallback_states_target_level_and_passes_grounding():
    fs = _sample_fact_sheet().model_copy(update={"metrics": [
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.0, status="pass"),
        MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=-0.1, status="pass"),
        MetricFact(metric_id="M9", display_name="AUROC", value=0.88, status="info"),
    ]})
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    resp = build_fallback_narrative(fs, refs, derived, reason="no_key", edition=3)

    benchmark = resp.conclusion.benchmark
    # S6: 8.1 은 개조식(';' 나열)이 아니라 같은 위치끼리 묶은 서술체 한 문장이다(작성 규칙 D37).
    assert "목표값이 참조 범위보다 낮은 지표는 Accuracy(목표 0), Recall(목표 -0.1)이다." in benchmark
    assert "AUROC(목표" not in benchmark  # 목표 없는 지표는 목표값 수준에 넣지 않는다
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, refs, derived, edition=3)).passed


def test_fallback_without_targets_has_no_target_sentence():
    fs = _sample_fact_sheet().model_copy(update={"metrics": [
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, status="info"),
    ]})
    refs = build_benchmark_refs("binary", fs.metrics, edition=3)
    resp = build_fallback_narrative(fs, refs, compute_derived(fs), reason="no_key", edition=3)
    assert "목표값 수준" not in resp.conclusion.benchmark


def test_whitelist_contains_target_values_from_refs():
    fs = _sample_fact_sheet()
    refs = [dict(_target_ref(0.77), target_value=0.77)]
    wl = build_number_whitelist(fs.model_copy(update={"metrics": []}), refs, {}, edition=3)
    assert "0.77" in wl


def test_prompt_explains_target_position():
    from app.narrative.legacy import build_user_prompt

    prompt = build_user_prompt({"verdict": "PASS"}, [], {})
    assert "target_position" in prompt and "target_value" in prompt
    assert "방향 해석은 하지 말고" in prompt


def test_benchmark_ref_out_carries_target_fields():
    from app.narrative.schemas import BenchmarkRefOut

    out = BenchmarkRefOut(**_target_ref(0.0))
    assert out.target_value == 0.0 and out.target_position == "below"
    # 옛 형태(목표 키 없음)도 받는다.
    legacy = {k: v for k, v in _target_ref(0.9).items() if not k.startswith("target_")}
    assert BenchmarkRefOut(**legacy).target_position is None


# ── 경계·확신 오류 비율(서술 설계 D30, B3) ─────────────────────────────────────

def _error_ratios(**over):
    from app.narrative.schemas import ErrorRatiosFact

    base = {
        "unit": "sample",
        "misclassified": 12,
        "boundary": {"count": 5, "ratio": 0.4167},
        "boundary_note": None,
        "confident": {"count": 2, "ratio": 0.1667},
        "boundary_margin": 0.1,
        "confident_level": 0.9,
        "threshold": 0.35,
    }
    base.update(over)
    return ErrorRatiosFact(**base)


_UNKNOWN = {"boundary": None, "boundary_note": "threshold_unknown", "threshold": None}


def _bare_fact_sheet(**over) -> FactSheet:
    """다른 사실에서 0.1·0.9 가 새어 들어오지 않는 최소 사실 시트(점수 0, 지표 없음)."""
    return FactSheet(verdict="INCONCLUSIVE", score=0.0, **over)


def test_whitelist_has_margin_and_level_only_with_error_ratios():
    """0.1·0.9 는 error_ratios 가 실렸을 때만 서술에 쓸 수 있다."""
    without = build_number_whitelist(_bare_fact_sheet(), [], {}, edition=3)
    assert "0.1" not in without and "0.9" not in without

    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=_error_ratios()), [], {}, edition=3)
    for tok in ("0.1", "0.9", "12", "5", "2", "0.4167", "41.7", "41.67", "0.1667", "16.7", "0.35"):
        assert tok in wl, tok


def test_whitelist_takes_per_label_error_thresholds():
    er = _error_ratios(unit="label_cell", threshold={"score_a": 0.42, "score_b": 0.61})
    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=er), [], {}, edition=3)
    assert "0.42" in wl and "0.61" in wl


def test_whitelist_skips_null_boundary():
    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=_error_ratios(**_UNKNOWN)), [], {}, edition=3)
    assert "0.4167" not in wl and "5" not in wl
    assert "0.1667" in wl


def test_fallback_states_boundary_and_confident_errors():
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios()})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)

    text = resp.interpretation.confusion_analysis
    assert (
        "오분류 12건 중 결정 경계와의 거리가 0.1 미만인 경계 오류는 5건(41.7%), "
        "틀린 쪽 확률이 0.9 이상인 확신 오류는 2건(16.7%)이다." in text
    )
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed


def test_fallback_says_boundary_not_computed_when_threshold_unknown():
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios(**_UNKNOWN)})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)

    text = resp.interpretation.confusion_analysis
    assert "결정 임계값을 알 수 없으므로 경계 오류는 계산하지 않았고" in text
    assert "확신 오류는 2건(16.7%)이다." in text
    assert "경계 오류는 5건" not in text
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed


def test_fallback_error_sentence_for_label_cells_and_zero_errors():
    cells = _error_ratios(unit="label_cell", threshold={"score_a": 0.5})
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": cells})
    text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).interpretation.confusion_analysis
    assert "틀린 레이블 칸 12칸 중" in text  # S6: 레이블 칸 단위는 '칸'(종전 '12건'은 7.1 의 '800칸'과 단위가 달랐다)

    zero = _error_ratios(misclassified=0, boundary={"count": 0, "ratio": None}, confident={"count": 0, "ratio": None})
    fs0 = _sample_fact_sheet().model_copy(update={"error_ratios": zero})
    text0 = build_fallback_narrative(fs0, [], compute_derived(fs0), "no_key", edition=3).interpretation.confusion_analysis
    assert "확률이 있는 오분류가 없어 경계 오류·확신 오류 비율은 산출하지 않았다." in text0


def test_fallback_without_error_ratios_is_unchanged():
    fs = _sample_fact_sheet()
    text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).interpretation.confusion_analysis
    assert "경계 오류" not in text and "확신 오류" not in text


def test_rule_11_follows_error_ratios():
    from app.narrative.legacy import build_system_prompt

    plain = build_system_prompt("internal", "binary", _sample_fact_sheet())
    known = build_system_prompt(
        "internal", "binary", _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios()})
    )
    unknown = build_system_prompt(
        "internal", "binary", _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios(**_UNKNOWN)})
    )
    cells = build_system_prompt(
        "internal", "multilabel",
        _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios(unit="label_cell")}),
    )

    assert "11. " not in plain and "error_ratios" not in plain
    assert "11. " in known and "confusion_analysis(7.1)" in known
    assert "threshold_unknown" not in known and "label_cell" not in known
    assert "경계 오류는 계산하지 않았다고" in unknown
    assert "틀린 레이블 칸" in cells


def test_error_ratios_schema_accepts_contract_shape_and_rejects_bad_values():
    from app.narrative.schemas import ErrorRatiosFact

    # 계약 §1.1 모양 그대로(평가 응답을 프론트가 옮겨 싣는다).
    req = NarrativeRequest(
        task_type=TaskType.binary,
        fact_sheet={
            "verdict": "PASS", "score": 100.0,
            "error_ratios": {
                "unit": "sample", "misclassified": 12,
                "boundary": None, "boundary_note": "threshold_unknown",
                "confident": {"count": 2, "ratio": 0.1667},
                "boundary_margin": 0.1, "confident_level": 0.9, "threshold": None,
            },
        },
    )
    assert req.fact_sheet.error_ratios.boundary is None
    # 옛 요청(키 없음)도 받는다.
    assert FactSheet(verdict="PASS", score=100.0).error_ratios is None

    with pytest.raises(ValidationError):
        _error_ratios(unit="row")
    with pytest.raises(ValidationError):
        _error_ratios(boundary_note="other")
    with pytest.raises(ValidationError):
        _error_ratios(confident={"count": -1, "ratio": 0.1})
    with pytest.raises(ValidationError):
        _error_ratios(confident={"count": 2, "ratio": 1.5})
    with pytest.raises(ValidationError):
        ErrorRatiosFact(unit="sample", misclassified=1)  # confident 필수
    with pytest.raises(ValidationError):
        _error_ratios(threshold={f"c{i}": 0.5 for i in range(257)})


# ── 신뢰구간이 목표값을 포함하는 지표(서술 설계 D29, B5) ────────────────────────

def _ci_metric(name="F1 Score", value=0.78, threshold=0.8, low=0.71, high=0.83, status="fail"):
    return MetricFact(
        metric_id="M4", display_name=name, value=value, threshold=threshold,
        status=status, ci_low=low, ci_high=high,
    )


@pytest.mark.parametrize(
    "threshold, contained",
    [
        (0.71, True),    # 하한과 같으면 포함(양 끝 포함)
        (0.83, True),    # 상한과 같으면 포함
        (0.8, True),
        (0.7099, False),
        (0.8301, False),
    ],
)
def test_ci_contains_target_boundaries(threshold, contained):
    fs = _bare_fact_sheet(metrics=[_ci_metric(threshold=threshold)])
    assert compute_derived(fs)["ci_contains_target"] == (["F1 Score"] if contained else [])


def test_ci_contains_target_skips_metrics_without_target_or_interval():
    fs = _bare_fact_sheet(metrics=[
        _ci_metric(name="Accuracy", threshold=None, status="info"),   # 목표 없음 → 판정 대상 아님
        _ci_metric(name="Recall", low=None, high=None),               # 구간 없음(skipped)
        _ci_metric(name="MCC", value=0.02, threshold=0.0, low=-0.05, high=0.09),  # 목표 0 도 목표다
    ])
    assert compute_derived(fs)["ci_contains_target"] == ["MCC"]


def test_ci_contains_target_absent_without_intervals():
    """구간이 하나도 없는 옛 사실 시트에는 키가 없다(프롬프트·폴백 그대로)."""
    assert "ci_contains_target" not in compute_derived(_sample_fact_sheet())


def test_whitelist_has_interval_bounds_and_level_only_with_intervals():
    without = build_number_whitelist(_bare_fact_sheet(metrics=[_ci_metric(low=None, high=None)]), [], {}, edition=3)
    assert "95" not in without and "0.71" not in without

    wl = build_number_whitelist(_bare_fact_sheet(metrics=[_ci_metric(low=-0.0512, high=0.8333)]), [], {}, edition=3)
    for tok in ("-0.0512", "0.8333", "0.95", "95"):
        assert tok in wl, tok


def test_fallback_states_metrics_whose_interval_contains_target():
    fs = _sample_fact_sheet().model_copy(update={"metrics": [
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass",
                   ci_low=0.9, ci_high=0.97),
        _ci_metric(name="Recall", value=0.7, threshold=0.8, low=0.63, high=0.81),
    ]})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)

    text = resp.conclusion.narrative
    assert (
        "95% 신뢰구간이 목표값을 포함하는 지표는 Recall(구간 0.63~0.81, 목표 0.8)로, "
        "표본 변동에 따라 해당 지표의 판정이 달라질 수 있다." in text
    )
    assert "Accuracy(구간" not in text
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed


def test_fallback_without_contained_targets_has_no_interval_sentence():
    fs = _sample_fact_sheet()
    assert "신뢰구간" not in build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).conclusion.narrative
    # S6: 구간이 모두 목표값과 떨어진 두 번째 갈래도 쓴다(필수 서술 항목 8.2 신뢰구간 해석 — 종전에는 아무 말이 없었다).
    outside = fs.model_copy(update={"metrics": [_ci_metric(threshold=0.9)]})
    narrative = build_fallback_narrative(outside, [], compute_derived(outside), "no_key", edition=3).conclusion.narrative
    assert "95% 신뢰구간이 모두 목표값과 떨어져 있어" in narrative
    assert "포함하는 지표" not in narrative


def test_rule_12_follows_intervals():
    from app.narrative.legacy import build_system_prompt

    plain = build_system_prompt("internal", "binary", _sample_fact_sheet())
    with_ci = build_system_prompt(
        "internal", "binary", _sample_fact_sheet().model_copy(update={"metrics": [_ci_metric()]})
    )
    assert "12. " not in plain and "ci_contains_target" not in plain
    assert "12. " in with_ci and "derived.ci_contains_target" in with_ci
    assert "conclusion.narrative(8.2)" in with_ci and "95% 신뢰구간" in with_ci
    assert "통계적 유의성 검정은 하지 않았다" in with_ci


def test_ci_fields_schema():
    req = NarrativeRequest(
        task_type=TaskType.binary,
        fact_sheet={
            "verdict": "PASS", "score": 100.0,
            "metrics": [{"metric_id": "M1", "display_name": "Accuracy", "value": 0.9, "threshold": 0.8,
                         "status": "pass", "ci_low": 0.85, "ci_high": 0.94}],
            "ci_method_note": "정확도는 Wilson 구간, 그 밖의 지표는 재표집(2,000회)으로 구했다.",
        },
    )
    assert req.fact_sheet.metrics[0].ci_low == 0.85
    # 옛 요청(키 없음)도 받는다.
    old = FactSheet(verdict="PASS", score=100.0, metrics=[MetricFact(
        metric_id="M1", display_name="Accuracy", value=0.9, status="info")])
    assert old.metrics[0].ci_low is None and old.ci_method_note is None
    with pytest.raises(ValidationError):
        FactSheet(verdict="PASS", score=100.0, ci_method_note="가" * 501)


# ── 아슬아슬 합격·임계값 조정 여지·불균형 영향(서술 설계 D12·D40·D20, B7) ─────────────────

def _near_metric(name="Recall", value=0.81, threshold=0.8, margin=0.01, near=True, status="pass"):
    return MetricFact(
        metric_id="M3", display_name=name, value=value, threshold=threshold,
        status=status, margin=margin, near_target=near,
    )


def _gain(**over):
    from app.narrative.schemas import F1ThresholdGainFact

    base = {"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37}
    base.update(over)
    return F1ThresholdGainFact(**base)


def _imbalance(**over):
    from app.narrative.schemas import ImbalanceEffectFact

    base = {
        "level": "affected", "imbalance_ratio": 3.2,
        "accuracy_minus_macro_f1": 0.12, "majority_minus_minority_f1": 0.18,
        "majority_class": "neg", "minority_class": "pos",
    }
    base.update(over)
    return ImbalanceEffectFact(**base)


def test_new_fact_fields_schema_accepts_contract_shape():
    req = NarrativeRequest(
        task_type=TaskType.binary,
        fact_sheet={
            "verdict": "PASS", "score": 100.0,
            "metrics": [{"metric_id": "M3", "display_name": "Recall", "value": 0.81, "threshold": 0.8,
                         "status": "pass", "margin": 0.01, "near_target": True}],
            "f1_threshold_gain": {"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37},
            "imbalance_effect": {"level": "affected", "imbalance_ratio": 3.2, "accuracy_minus_macro_f1": None,
                                 "majority_minus_minority_f1": -0.05, "majority_class": "2", "minority_class": "0"},
        },
    )
    fs = req.fact_sheet
    assert fs.metrics[0].near_target is True and fs.metrics[0].margin == 0.01
    assert fs.f1_threshold_gain.gain == 0.1409
    assert fs.imbalance_effect.accuracy_minus_macro_f1 is None
    # 옛 요청(키 없음)도 받는다.
    old = FactSheet(verdict="PASS", score=100.0, metrics=[MetricFact(
        metric_id="M1", display_name="Accuracy", value=0.9, status="info")])
    assert old.metrics[0].margin is None and old.metrics[0].near_target is False
    assert old.f1_threshold_gain is None and old.imbalance_effect is None
    with pytest.raises(ValidationError):
        _imbalance(level="severe")
    with pytest.raises(ValidationError):
        _gain(gain=-0.01)


@pytest.mark.parametrize("gain, notable", [(0.1, True), (0.0999, False), (0.1409, True), (0.0, False)])
def test_threshold_tuning_boundary(gain, notable):
    """gain ≥ 0.10 (경계 포함) — 프론트 문제 목록 threshold_tuning 과 같은 기준."""
    fs = _bare_fact_sheet(f1_threshold_gain=_gain(gain=gain))
    assert compute_derived(fs)["threshold_tuning"] is notable


@pytest.mark.parametrize("gain, notable", [(0.1, False), (0.0999, True)])
def test_threshold_tuning_reads_engine_notable(gain, notable):
    """엔진이 반올림 전 값으로 정한 notable 이 실리면 그것을 따른다 — 실제 0.09996 은 4자리로 0.1 이지만 부적합이 아니다."""
    fs = _bare_fact_sheet(f1_threshold_gain=_gain(gain=gain, notable=notable))
    assert compute_derived(fs)["threshold_tuning"] is notable


def test_threshold_tuning_absent_without_gain():
    assert "threshold_tuning" not in compute_derived(_sample_fact_sheet())


def test_whitelist_new_numbers_only_with_facts():
    """margin·3%·gain 네 값·0.1·불균형 값·1.5 는 해당 사실이 실렸을 때만 허용한다."""
    without = build_number_whitelist(_bare_fact_sheet(), [], {}, edition=3)
    for tok in ("0.1", "1.5", "3", "0.03", "0.37", "3.2"):
        assert tok not in without, tok

    near = build_number_whitelist(_bare_fact_sheet(metrics=[_near_metric(margin=0.012)]), [], {}, edition=3)
    for tok in ("0.012", "0.03", "3"):
        assert tok in near, tok
    # near_target 이 아니면 3% 는 허용하지 않는다(margin 은 사실이라 허용).
    not_near = build_number_whitelist(_bare_fact_sheet(metrics=[_near_metric(margin=0.2, near=False)]), [], {}, edition=3)
    assert "0.2" in not_near and "0.03" not in not_near

    gain = build_number_whitelist(_bare_fact_sheet(f1_threshold_gain=_gain()), [], {}, edition=3)
    for tok in ("0.8421", "0.7012", "0.1409", "0.37", "0.1"):
        assert tok in gain, tok
    assert "1.5" not in gain

    ie = build_number_whitelist(_bare_fact_sheet(imbalance_effect=_imbalance(
        majority_minus_minority_f1=-0.05, majority_class="2", minority_class="7")), [], {}, edition=3)
    for tok in ("3.2", "0.12", "-0.05", "1.5", "0.1", "2", "7"):
        assert tok in ie, tok


def test_fallback_risks_mention_near_target_pass_metrics():
    fs = FactSheet(verdict="PASS", score=100.0, metrics=[
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass",
                   margin=0.09, near_target=False),
        _near_metric(),
    ])
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)

    risks = resp.conclusion.risks
    assert (
        "목표값과의 차이가 목표값의 3% 미만인 아슬아슬 합격 지표는 Recall(0.81, 목표 0.8, 차이 0.01)로, "
        "표본이 달라지면 목표에 미달할 위험이 있다." in risks
    )
    assert "Accuracy(" not in risks
    assert "식별된 목표 미달 위험은 없다" not in risks
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed
    # 합격 성적서의 이 서술이 판정 모순으로 걸리지 않는다.
    from app.narrative.grounding import find_verdict_contradictions
    assert find_verdict_contradictions(_all_texts(resp), "PASS") == []


def test_fallback_risks_skip_near_target_fail_and_absent_flags():
    """근소 미달(near_target 이지만 fail)은 아슬아슬 '합격'이 아니다. 키가 없으면 종전 문장 그대로."""
    fs = _sample_fact_sheet()
    assert build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).conclusion.risks == (
        "목표에 미달한 지표가 있어 해당 영역의 성능 저하 위험이 존재한다(목표 미달 지표: Recall(0.7, 목표 0.8))."
    )
    near_fail = _bare_fact_sheet(metrics=[_near_metric(value=0.79, margin=-0.01, status="fail")])
    assert "아슬아슬" not in build_fallback_narrative(
        near_fail, [], compute_derived(near_fail), "no_key", edition=3).conclusion.risks


def test_fallback_metric_numbers_use_report_digits():
    """S10: 규칙 폴백 8절이 "Recall(0.5473684210526316, 기준 0.8)"처럼 반올림 안 한 수를 찍었다.
    성적서처럼 셋째 자리까지만 쓴다(끝의 0 은 뺀다). 자리마다 따로 고정한다 — 미달·아슬아슬·신뢰구간."""
    raw = 0.5473684210526316
    failed = _bare_fact_sheet(metrics=[MetricFact(
        metric_id="M3", display_name="Recall", value=raw, threshold=0.8, status="fail")])
    resp = build_fallback_narrative(failed, [], compute_derived(failed), "no_key", edition=3)
    assert "Recall(0.547, 목표 0.8)" in resp.conclusion.risks
    assert "0.5473" not in " ".join(_all_texts(resp))
    assert verify_grounding(_all_texts(resp), build_number_whitelist(failed, [], compute_derived(failed), edition=3)).passed

    near = _bare_fact_sheet(metrics=[_near_metric(value=0.80789, threshold=0.8, margin=0.00789)])
    near_risks = build_fallback_narrative(near, [], compute_derived(near), "no_key", edition=3).conclusion.risks
    assert "Recall(0.808, 목표 0.8, 차이 0.008)" in near_risks

    ci = _bare_fact_sheet(metrics=[_ci_metric(value=0.78123, threshold=0.8, low=0.712345, high=0.83456)])
    ci_resp = build_fallback_narrative(ci, [], compute_derived(ci), "no_key", edition=3)
    assert "F1 Score(구간 0.712~0.835, 목표 0.8)" in " ".join(_all_texts(ci_resp))
    assert "F1 Score(0.781, 목표 0.8)" in ci_resp.conclusion.risks


def test_fallback_long_threshold_is_rounded_at_every_site():
    """S10 검토: 목표값이 0.8 처럼 짧으면 반올림을 빼도 테스트가 통과했다. 긴 목표값(2/3)으로
    미달·아슬아슬·신뢰구간 세 자리의 목표값을 따로 고정한다."""
    t = 2 / 3
    failed = _bare_fact_sheet(metrics=[MetricFact(
        metric_id="M3", display_name="Recall", value=0.5, threshold=t, status="fail")])
    risks = build_fallback_narrative(failed, [], compute_derived(failed), "no_key", edition=3).conclusion.risks
    assert "Recall(0.5, 목표 0.667)" in risks

    near = _bare_fact_sheet(metrics=[_near_metric(value=0.68, threshold=t, margin=0.68 - t)])
    near_risks = build_fallback_narrative(near, [], compute_derived(near), "no_key", edition=3).conclusion.risks
    assert "Recall(0.68, 목표 0.667, 차이 0.013)" in near_risks

    ci = _bare_fact_sheet(metrics=[_ci_metric(value=0.6, threshold=t, low=0.55, high=0.7)])
    ci_texts = " ".join(_all_texts(build_fallback_narrative(ci, [], compute_derived(ci), "no_key", edition=3)))
    assert "F1 Score(구간 0.55~0.7, 목표 0.667)" in ci_texts
    assert "0.6666" not in ci_texts


def test_fallback_risks_state_threshold_gain_only_when_notable():
    fs = _sample_fact_sheet().model_copy(update={"f1_threshold_gain": _gain()})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)
    risks = resp.conclusion.risks
    # S6: 숫자(임계값·F1·향상폭)는 결론 목록대로 8.2(임계값 조정 여지)에, 8.3 은 숫자 없이 위험만 — 두 칸에 같은 숫자를
    # 되풀이하지 않는다. 0.37 은 받침 ㄹ(칠)이라 '로'.
    narrative = resp.conclusion.narrative
    assert "결정 임계값을 0.37로 바꾸면 F1 이 0.7012에서 0.8421까지 오를 수 있다." in narrative
    assert "이 데이터의 최대폭이므로 별도 데이터에서 확인해야 한다." in narrative
    assert "현재 결정 임계값이 최적이 아닐 수 있어" in risks and "0.37" not in risks
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed

    small = _sample_fact_sheet().model_copy(update={"f1_threshold_gain": _gain(gain=0.0999)})
    small_resp = build_fallback_narrative(small, [], compute_derived(small), "no_key", edition=3)
    assert "최대폭" not in small_resp.conclusion.narrative and "최적이 아닐" not in small_resp.conclusion.risks


# 서술 라운드 S1: 문장을 기준 이상인 차이만 근거로 쓰게 고쳤다(점검 BE-02). 엔진이 낼 수 없는 조합
# (no_distortion 인데 두 차이 0.12·0.18, mild 인데 비율 3.2)은 엔진이 실제로 낼 값으로 바꿨다.
@pytest.mark.parametrize("level, values, expected, absent", [
    ("affected", {},
     "불균형 비율(3.2)이 1.5 이상이어서 불균형이 있고, 정확도와 매크로 F1 의 차이(0.12)와 다수 클래스(neg)와 "
     "소수 클래스(pos)의 F1 차이(0.18)가 기준(0.1) 이상이어서 불균형에 따른 성능 지표 왜곡이 관찰되었다.",
     "왜곡은 관찰되지 않았다"),
    ("no_distortion", {"accuracy_minus_macro_f1": 0.05, "majority_minus_minority_f1": 0.03},
     "불균형 비율(3.2)이 1.5 이상이어서 불균형이 있으나, 불균형에 의한 성능 왜곡은 관찰되지 않았다. "
     "정확도와 매크로 F1 의 차이(0.05)는 기준(0.1) 미만이다. 다수 클래스(neg)와 소수 클래스(pos)의 F1 "
     "차이(0.03)는 기준(0.1) 미만이다.",
     "영향을 준"),
    ("mild", {"imbalance_ratio": 1.3}, "불균형 비율(1.3)이 1.5 미만이어서 불균형은 경미하다.", "불균형이 있"),
])
def test_fallback_distribution_follows_imbalance_level(level, values, expected, absent):
    """7.2 의 결론은 엔진이 낸 level 로만 쓴다 — distribution.imbalance_ratio(1.857)로 다시 판정하지 않는다."""
    fs = _sample_fact_sheet().model_copy(update={"imbalance_effect": _imbalance(level=level, **values)})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key", edition=3)
    text = resp.interpretation.distribution_analysis
    assert expected in text
    assert absent not in text
    assert "1.857" not in text
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived, edition=3)).passed


def test_fallback_distribution_multilabel_uses_label_gap_only():
    ie = _imbalance(accuracy_minus_macro_f1=None, majority_minus_minority_f1=-0.04, level="no_distortion")
    fs = _sample_fact_sheet().model_copy(update={"imbalance_effect": ie})
    text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).interpretation.distribution_analysis
    assert "다수 레이블(neg)과 소수 레이블(pos)의 F1 차이(-0.04)는 음수로 소수 레이블의 F1 이 더 높다." in text
    assert "매크로 F1" not in text


def test_fallback_distribution_without_imbalance_effect_is_neutral():
    """키가 없으면(옛 프론트) 비율만 적고 1.5 판정을 하지 않는다 — 종전의 'IR ≤ 1.5 = 균형' 규칙은 없앴다."""
    for ir in (1.4, 1.5, 1.857):
        fs = _sample_fact_sheet().model_copy(update={"distribution": DistributionFact(
            class_distribution={"0": 130, "1": 70}, imbalance_ratio=ir)})
        text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key", edition=3).interpretation.distribution_analysis
        assert text.endswith(f" 불균형 비율은 {ir}이다."), text
        assert "균형 상태" not in text and "경미" not in text


def test_rules_13_to_15_follow_facts():
    from app.narrative.legacy import build_system_prompt

    plain = build_system_prompt("internal", "binary", _sample_fact_sheet())
    for n in ("13. ", "14. ", "15. "):
        assert n not in plain
    assert "near_target" not in plain and "f1_threshold_gain" not in plain and "imbalance_effect" not in plain

    near = build_system_prompt("internal", "binary", _bare_fact_sheet(metrics=[_near_metric()]))
    assert "13. " in near and "conclusion.risks(8.3)" in near and "아슬아슬 합격" in near
    near_fail = build_system_prompt(
        "internal", "binary", _bare_fact_sheet(metrics=[_near_metric(status="fail", margin=-0.01)]))
    assert "13. " not in near_fail

    # 규칙 14 는 compute_derived 가 한 번 정한 threshold_tuning 을 읽는다(S4b — 따로 판정하지 않음).
    gain_fs = _bare_fact_sheet(f1_threshold_gain=_gain())
    gain = build_system_prompt("internal", "binary", gain_fs, compute_derived(gain_fs))
    assert "14. " in gain and "이 데이터에서 오를 수 있는 최대폭" in gain and "별도 데이터에서 확인" in gain
    assert "conclusion.narrative(8.2)" in gain and "conclusion.risks(8.3)" in gain
    small_fs = _bare_fact_sheet(f1_threshold_gain=_gain(gain=0.0999))
    assert "14. " not in build_system_prompt("internal", "binary", small_fs, compute_derived(small_fs))
    # 엔진 notable 이 판정이다 — 4자리로 0.1 이어도 notable=False 면 규칙이 없다(0.09996).
    edge_fs = _bare_fact_sheet(f1_threshold_gain=_gain(gain=0.1, notable=False))
    assert "14. " not in build_system_prompt("internal", "binary", edge_fs, compute_derived(edge_fs))
    assert "14. " not in build_system_prompt("internal", "binary", gain_fs)  # derived 없이 부르면 붙지 않는다

    for level, phrase in (("affected", "성능 수치에 영향을 주었다고"),
                          ("no_distortion", "성능 왜곡은 관찰되지 않았다"),
                          ("mild", "불균형이 경미하다")):
        prompt = build_system_prompt("internal", "binary", _bare_fact_sheet(imbalance_effect=_imbalance(level=level)))
        assert "15. " in prompt and "distribution_analysis(7.2)" in prompt and phrase in prompt, level
        others = [p for lv, p in (("affected", "성능 수치에 영향을 주었다고"),
                                  ("no_distortion", "성능 왜곡은 관찰되지 않았다"),
                                  ("mild", "불균형이 경미하다")) if lv != level]
        for other in others:
            assert other not in prompt, (level, other)


# ── 세션 3 검토 반영: 소수점 절 번호·신뢰구간 방식 문장·margin 크기 ─────────────────────

@pytest.mark.parametrize("text", [
    "8.3절에서 지적한 위험을 9절에서 다룬다.",
    "구간이 목표값을 포함한다(8.2).",
    "9.2절 권고를 따른다.",
    "7.1 절의 오류 분석과 같다.",
    "8절 8.3 의 내용과 같다.",
    # 세션 3 검토 2 — 절 번호 나열과 문장 끝 "절 8.3."
    "7.1·7.2절의 오류 분석을 따른다.",
    "8.2·8.3절에서 다룬 위험이다.",
    "8.2와 8.3절에서 다룬 위험이다.",
    "구간과 아슬아슬 판정을 함께 본다(8.2, 8.3).",
    "오류 분석과 같다(7.1·7.2).",
    "자세한 내용은 절 8.3.",
    "환경과 신뢰구간은 6.1·6.4절, 지표 정의는 (7.3, 9.1) 에 있다.",
    # 2026-10-19 Claude 실호출 — 9절 칸 맨 앞에 소절 머리표를 붙여 썼다.
    "9.1 데이터 측면에서는 클래스 불균형 영향(P1)이 지목되었다.",
    "  9.2 모델 측면에서 지목된 문제는 없다.",
    "9.1에서 지목된 문제는 P1(불균형 영향 있음)이다.",
    "9.2에서 지목된 문제는 없다.",
])
def test_dotted_section_references_are_not_numbers(text):
    """"8.3절"·"(8.2)" 는 절 번호다 — 종전에는 "3절"만 지우고 "8." 을 환각 수치로 잡았다."""
    from app.narrative.grounding import has_number_token

    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    g = verify_grounding([text], wl)
    assert g.passed, g.violations
    assert has_number_token(text) is False


@pytest.mark.parametrize("text, fake", [
    ("8.3절에서 지적한 대로 정확도는 0.82 이다.", "0.82"),
    ("정확도는 (0.82) 이다.", "(0.82)"),
    ("오류율이 8.2% 로 8.2절과 같다.", "8.2%"),
    ("비용을 0.3 절감한다.", "0.3"),
    # 세션 3 검토 2 — 나열 꼴은 실제 절 번호끼리일 때만 지운다.
    ("구간은 (0.82, 0.91) 이다.", "0.82"),
    ("구간은 (0.82, 0.91) 이다.", "0.91"),
    ("8.2·8.3절에서 본 정확도는 0.82 이다.", "0.82"),
    ("오류율이 8.2% 로 7.1·7.2절과 같다.", "8.2%"),
    ("값은 (8.2, 0.82) 이다.", "0.82"),
    ("절 8.3.5 를 본다.", "8.3"),
    ("기준 1.5·0.82절과 비교한다.", "1.5"),
    # 소절 머리표 규칙은 글 맨 앞 + 실제 절 번호 + 띄어쓰기 + 한글일 때만이다.
    ("정확도는 9.1 수준이다.", "9.1"),
    ("0.82 수준의 정확도다.", "0.82"),
    ("9.1% 의 오류율이다.", "9.1%"),
    ("8.4 수준의 비율이다.", "8.4"),
    ("9.1로 측정되었다.", "9.1"),
])
def test_dotted_section_rule_does_not_whitelist_fake_values(text, fake):
    """절 번호 꼴만 지운다 — 지표값 꼴(0.82)·백분율(8.2%)·'절감' 앞 숫자는 그대로 검증한다."""
    from app.narrative.grounding import has_number_token

    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs), edition=3)
    g = verify_grounding([text], wl)
    assert not g.passed
    assert any(fake.strip("()") in v for v in g.violations), g.violations
    assert has_number_token(text) is True


def test_whitelist_includes_numbers_in_ci_method_note():
    """6.0 신뢰구간 방식 한 줄의 숫자("2,000회")를 서술이 옮겨 적어도 환각이 아니다."""
    text = ["신뢰구간은 재표집 2,000회로 구했다."]
    bare = build_number_whitelist(_bare_fact_sheet(), [], {}, edition=3)
    assert not verify_grounding(text, bare).passed

    fs = _bare_fact_sheet(ci_method_note="정확도는 Wilson 구간, 그 밖의 지표는 재표집(2,000회)으로 구했다.")
    assert verify_grounding(text, build_number_whitelist(fs, [], {}, edition=3)).passed
    # 없으면(None) 건너뛴다.
    build_number_whitelist(_bare_fact_sheet(ci_method_note=None), [], {}, edition=3)


def test_whitelist_includes_absolute_margin():
    """margin 이 음수여도 "목표값보다 0.0288 낮다"처럼 크기만 적는 서술을 허용한다."""
    fs = _bare_fact_sheet(metrics=[_near_metric(value=0.7712, threshold=0.8, margin=-0.0288,
                                                 near=False, status="fail")])
    wl = build_number_whitelist(fs, [], {}, edition=3)
    assert "-0.0288" in wl and "0.0288" in wl
    assert verify_grounding(["Recall 은 목표값보다 0.0288 낮다."], wl).passed
