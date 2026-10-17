"""
test_narrator.py — LLM 서술 모듈의 환각 방어선(grounding) + 규칙 폴백 단위 테스트.
LLM·API 키 없이 검증 가능한 순수 함수 대상.
"""
import pytest
from pydantic import ValidationError

from app.core.schemas import TaskType
from app.narrative.schemas import ConfusionFact, DistributionFact, FactSheet, MetricFact, NarrativeRequest
from app.narrative.service import generate_narrative
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
    refs = build_benchmark_refs("binary", fs.metrics)
    wl = build_number_whitelist(fs, refs, derived)

    resp = build_fallback_narrative(fs, refs, derived, reason="no_key")
    grounding = verify_grounding(_all_texts(resp), wl)

    assert grounding.passed, f"폴백이 화이트리스트 밖 숫자 사용: {grounding.violations}"
    assert grounding.checked > 0, "검사된 숫자가 없음(템플릿이 비었는지 확인)"
    assert resp.conclusion.verdict == "CONDITIONAL_PASS"
    assert resp.meta.source == "fallback"


def test_grounding_catches_hallucination():
    """fact_sheet 에 없는 숫자(환각)를 주입하면 검증이 위반으로 잡아야 한다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics)
    wl = build_number_whitelist(fs, refs, derived)

    hallucinated = ["정확도는 무려 99.7%에 달하며 P99 지연시간은 41.2ms이다."]
    grounding = verify_grounding(hallucinated, wl)

    assert not grounding.passed
    assert any("99.7" in v for v in grounding.violations)
    assert any("41.2" in v for v in grounding.violations)


def test_exempt_tokens_not_flagged():
    """절 번호/ISO 연도 같은 면제 토큰은 위반으로 잡지 않는다."""
    fs = _sample_fact_sheet()
    derived = compute_derived(fs)
    wl = build_number_whitelist(fs, [], derived)
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
    wl = build_number_whitelist(fs, [], compute_derived(fs))
    g = verify_grounding(["정확도는 100%이며 제외 0건, 신뢰도 1.0, 총 2,022건이다."], wl)
    assert not g.passed
    assert "100%" in g.violations and "0" in g.violations and "2,022" in g.violations


def test_genuine_perfect_and_zero_pass():
    """실측 100%(=1.0)·0건이 fact_sheet 근거이면 통과한다(과차단 방지)."""
    fs = FactSheet(
        n_samples=100, dropped_rows=0, verdict="PASS", score=100.0,
        metrics=[MetricFact(metric_id="M1", display_name="Accuracy", value=1.0, threshold=0.9, status="pass")],
    )
    wl = build_number_whitelist(fs, [], compute_derived(fs))
    g = verify_grounding(["정확도 100%, 제외 0건, 점수 1.0"], wl)
    assert g.passed, g.violations


def test_metric_name_and_ordinal_not_flagged():
    """F1·P99 식별자와 제N종·N절 문맥 토큰은 오탐하지 않는다(D3a)."""
    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs))
    g = verify_grounding(["F1 Score와 P99 지연, 제1종·제2종 오류, 7절 참조."], wl)
    assert g.passed, g.violations


def test_grounding_covers_recommendation_category():
    """recommendations[].category 의 환각 숫자도 전수 수집으로 검증된다(D3b)."""
    fs = _no_special_fact_sheet()
    wl = build_number_whitelist(fs, [], compute_derived(fs))
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

async def test_internal_grounding_violation_falls_back(make_fake_openai_client):
    """report_purpose=internal 이어도 grounding 위반 시 규칙 폴백으로 강등(fail-open 제거, D3c)."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "재현율은 0.777로 낮다.", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_openai_client(llm_json)
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="internal", fact_sheet=fs)
    resp = await generate_narrative(client, req)
    assert resp.meta.source == "fallback"
    assert resp.meta.reason == "grounding_failed"
    assert resp.meta.grounding.passed is False
    # 폴백 산문은 fact_sheet 근거만 사용하므로 자체 grounding 통과
    wl = build_number_whitelist(fs, build_benchmark_refs("binary", fs.metrics), compute_derived(fs))
    assert verify_grounding(_all_texts(resp), wl).passed


async def test_clean_llm_response_source_llm(make_fake_openai_client):
    """근거 있는 정상 응답은 source=llm, verdict 는 서버값으로 강제."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "정확도 0.94로 양호.", "distribution_analysis": ""},
        "conclusion": {"verdict": "누가봐도이상함", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_openai_client(llm_json)
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
        "multilabel", [MetricFact(metric_id="M15", display_name="Hamming Loss", value=0.02, threshold=0.1, status="pass")]
    )[0]
    bad = build_benchmark_refs(
        "multilabel", [MetricFact(metric_id="M15", display_name="Hamming Loss", value=0.30, threshold=0.1, status="fail")]
    )[0]
    assert good["direction"] == "lower" and good["quality"] == "better"
    assert bad["quality"] == "worse"


def test_benchmark_direction_higher_is_better():
    """Accuracy(높을수록 좋음): 범위 위=우수 (D7[3])."""
    ref = build_benchmark_refs(
        "binary", [MetricFact(metric_id="M1", display_name="Accuracy", value=0.95, threshold=0.85, status="pass")]
    )[0]
    assert ref["direction"] == "higher" and ref["quality"] == "better"


async def test_assembly_error_falls_back(make_fake_openai_client):
    """스키마 불일치(조립 불가) 응답은 500 대신 규칙 폴백(assembly_error, D7[1])."""
    fs = _no_special_fact_sheet()
    llm_json = {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": ["not-a-dict"],  # RecommendationOut(**r) 조립 실패 유발
    }
    client = make_fake_openai_client(llm_json)
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
    benchmark_refs = build_benchmark_refs("binary", fs.metrics)
    resp = build_fallback_narrative(fs, benchmark_refs, compute_derived(fs), reason="no_key")
    assert resp.conclusion.verdict == "INCONCLUSIVE"
    assert "판정 불가" in resp.conclusion.narrative


def test_client_input_numbers_are_allowed():
    fs = _sample_fact_sheet()
    benchmark_refs = build_benchmark_refs("binary", fs.metrics)
    derived = compute_derived(fs)
    client_inputs = {
        "training_data": {"학습 샘플 수": "12,000개", "수집 기간": "2023-01 ~ 2024-12"},
        "ground_truth": {"라벨러 수": "3명", "라벨 일치도": "Cohen의 카파 0.71"},
    }
    text = ["학습 데이터 12,000개, 라벨러 3명, 카파 0.71 로 보고되었다."]

    without = verify_grounding(text, build_number_whitelist(fs, benchmark_refs, derived))
    assert not without.passed

    with_inputs = build_number_whitelist(fs, benchmark_refs, derived, client_inputs)
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


def test_client_inputs_reach_the_prompt():
    from app.narrative.prompt import build_user_prompt

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
    wl = build_number_whitelist(fs, [], derived)

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
    from app.narrative.prompt import build_system_prompt

    assert "다중 클래스 분류" in build_system_prompt("internal", "multiclass")
    assert "멀티레이블 분류" in build_system_prompt("internal", "multilabel")


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
    wl = build_number_whitelist(fs, [], compute_derived(fs))

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
    wl = build_number_whitelist(fs, [], compute_derived(fs))

    assert verify_grounding(["레이블 a 는 0.4, b 는 0.65 로 판정했다."], wl).passed


async def test_benchmark_refs_come_back_with_the_narrative():
    """8.1 에 참조 범위를 숫자로 찍도록 비교값을 응답에 싣는다(재료 목록 9절). 폴백이어도 같다."""
    req = NarrativeRequest(task_type=TaskType.binary, fact_sheet=_sample_fact_sheet())
    resp = await generate_narrative(None, req)

    expected = build_benchmark_refs("binary", req.fact_sheet.metrics)
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
    wl = build_number_whitelist(fs, [], compute_derived(fs))

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
    resp = build_fallback_narrative(fs, [], d, "no_key")
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
    from app.narrative.prompt import build_system_prompt

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


async def test_service_sends_the_method_specific_rule(make_fake_openai_client):
    llm_json = {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": [],
    }
    client = make_fake_openai_client(llm_json)
    req = NarrativeRequest(
        task_type=TaskType.multiclass, report_purpose="internal", fact_sheet=_with_derivation("argmax")
    )
    await generate_narrative(client, req)

    system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "가장 큰 클래스" in system
    assert "임계값을 적용" not in system


# ── 목표값 수준(서술 설계 D31) ─────────────────────────────────────────────────
#
# 참조 범위와 목표값이 만나는 곳: 목표를 범위보다 낮게/안에/높게 잡았는지를 단순 수치 위치로
# 싣는다(방향 반영 없음). 목표값 0·음수도 실제 목표다 — None 일 때만 '목표 없음'.

def _target_ref(threshold, value=0.88, name="Accuracy", task="binary"):
    m = MetricFact(metric_id="M1", display_name=name, value=value, threshold=threshold, status="info")
    return build_benchmark_refs(task, [m])[0]


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
    refs = build_benchmark_refs("binary", fs.metrics)
    resp = build_fallback_narrative(fs, refs, derived, reason="no_key")

    benchmark = resp.conclusion.benchmark
    assert "목표값 수준은 Accuracy 0.0(참조 범위보다 낮음), Recall -0.1(참조 범위보다 낮음)이다." in benchmark
    assert "AUROC 0.88(" not in benchmark  # 목표 없는 지표는 목표값 수준에 넣지 않는다
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, refs, derived)).passed


def test_fallback_without_targets_has_no_target_sentence():
    fs = _sample_fact_sheet().model_copy(update={"metrics": [
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, status="info"),
    ]})
    refs = build_benchmark_refs("binary", fs.metrics)
    resp = build_fallback_narrative(fs, refs, compute_derived(fs), reason="no_key")
    assert "목표값 수준" not in resp.conclusion.benchmark


def test_whitelist_contains_target_values_from_refs():
    fs = _sample_fact_sheet()
    refs = [dict(_target_ref(0.77), target_value=0.77)]
    wl = build_number_whitelist(fs.model_copy(update={"metrics": []}), refs, {})
    assert "0.77" in wl


def test_prompt_explains_target_position():
    from app.narrative.prompt import build_user_prompt

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
    without = build_number_whitelist(_bare_fact_sheet(), [], {})
    assert "0.1" not in without and "0.9" not in without

    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=_error_ratios()), [], {})
    for tok in ("0.1", "0.9", "12", "5", "2", "0.4167", "41.7", "41.67", "0.1667", "16.7", "0.35"):
        assert tok in wl, tok


def test_whitelist_takes_per_label_error_thresholds():
    er = _error_ratios(unit="label_cell", threshold={"score_a": 0.42, "score_b": 0.61})
    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=er), [], {})
    assert "0.42" in wl and "0.61" in wl


def test_whitelist_skips_null_boundary():
    wl = build_number_whitelist(_bare_fact_sheet(error_ratios=_error_ratios(**_UNKNOWN)), [], {})
    assert "0.4167" not in wl and "5" not in wl
    assert "0.1667" in wl


def test_fallback_states_boundary_and_confident_errors():
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios()})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key")

    text = resp.interpretation.confusion_analysis
    assert (
        "오분류 12건 중 결정 경계와의 거리가 0.1 미만인 경계 오류는 5건(41.7%), "
        "틀린 쪽 확률이 0.9 이상인 확신 오류는 2건(16.7%)이다." in text
    )
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived)).passed


def test_fallback_says_boundary_not_computed_when_threshold_unknown():
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": _error_ratios(**_UNKNOWN)})
    derived = compute_derived(fs)
    resp = build_fallback_narrative(fs, [], derived, reason="no_key")

    text = resp.interpretation.confusion_analysis
    assert "결정 임계값을 알 수 없으므로 경계 오류는 계산하지 않았고" in text
    assert "확신 오류는 2건(16.7%)이다." in text
    assert "경계 오류는 5건" not in text
    assert verify_grounding(_all_texts(resp), build_number_whitelist(fs, [], derived)).passed


def test_fallback_error_sentence_for_label_cells_and_zero_errors():
    cells = _error_ratios(unit="label_cell", threshold={"score_a": 0.5})
    fs = _sample_fact_sheet().model_copy(update={"error_ratios": cells})
    text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key").interpretation.confusion_analysis
    assert "틀린 레이블 칸 12건 중" in text

    zero = _error_ratios(misclassified=0, boundary={"count": 0, "ratio": None}, confident={"count": 0, "ratio": None})
    fs0 = _sample_fact_sheet().model_copy(update={"error_ratios": zero})
    text0 = build_fallback_narrative(fs0, [], compute_derived(fs0), "no_key").interpretation.confusion_analysis
    assert "확률이 있는 오분류가 없어 경계 오류·확신 오류 비율은 산출하지 않았다." in text0


def test_fallback_without_error_ratios_is_unchanged():
    fs = _sample_fact_sheet()
    text = build_fallback_narrative(fs, [], compute_derived(fs), "no_key").interpretation.confusion_analysis
    assert "경계 오류" not in text and "확신 오류" not in text


def test_rule_11_follows_error_ratios():
    from app.narrative.prompt import build_system_prompt

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
