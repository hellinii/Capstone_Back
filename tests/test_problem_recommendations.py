"""tests/test_problem_recommendations.py — 9절 권고를 문제 목록에 묶는 규칙(서술 설계 D22).

계약(세션 3 계약서 §2.2·§3):
- 사실 시트의 problems(ProblemFact, 프론트 problemList.ts 가 만든다)가 권고의 유일한 근거다.
- LLM 응답 스키마의 problem_id 는 그 요청의 번호 enum. 문제가 없으면 권고 배열은 maxItems 0.
- 서버 후처리: 모르는·빈 번호의 권고는 버린다, 5개 자르기 없음, 권고 없는 문제는 종류별 템플릿으로 채움,
  문제 번호 순서로 정렬, priority 는 문제의 것으로 강제, 기대 효과에 숫자가 있으면 전체 폴백(impact_number).
- 폴백 권고 = 종류별 템플릿(기대 효과에 숫자 없음). 문제가 없으면 권고도 없다.
"""
import pytest
from pydantic import ValidationError

from app.core.schemas import TaskType
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.fallback import _PROBLEM_TEMPLATES, build_fallback_narrative, problem_recommendation
from app.narrative.grounding import (
    _collect_grounding_texts,
    build_number_whitelist,
    has_number_token,
    verify_grounding,
)
from app.narrative.prompt import build_response_schema, build_user_prompt
from app.narrative.schemas import (
    MAX_PROBLEM_EVIDENCE_CHARS,
    MAX_PROBLEMS,
    FactSheet,
    MetricFact,
    NarrativeRequest,
    ProblemFact,
)
from app.narrative.service import generate_narrative

_SECTIONS = {
    "core_fail": ("8.2", "9.2"),
    "inconclusive": ("8.2", "9.2"),
    "noncore_fail": ("8.2", "9.2"),
    "core_near_target": ("8.3", "9.2"),
    "threshold_tuning": ("8.3", "9.2"),
    "imbalance_effect": ("7.2", "9.1"),
    "weakest_class": ("7.1·8.3", "9.1"),
    "distribution_shift": ("7.2", "9.1"),
    "label_caution": ("7.2", "9.1"),
}


def _problem(pid: str, kind: str, priority: str, title: str = "", evidence=None, note=None) -> ProblemFact:
    source, recommend = _SECTIONS[kind]
    return ProblemFact(
        id=pid, kind=kind, priority=priority, title=title or kind,
        source_section=source, recommend_section=recommend,
        evidence=evidence or {}, note=note,
    )


def _problems() -> list[ProblemFact]:
    return [
        _problem("P1", "core_fail", "HIGH", "핵심 지표 미달 (Recall)",
                 {"Recall": {"value": 0.7, "target": 0.8}}),
        _problem("P2", "threshold_tuning", "MEDIUM", "임계값 부적합",
                 {"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37}),
        _problem("P3", "imbalance_effect", "MEDIUM", "불균형 영향 있음 (다수 0 · 소수 1)",
                 {"imbalance_ratio": 3.2, "majority_class": "0", "minority_class": "1"},
                 note="의뢰자가 편향 감소 노력을 '수행 안 함'으로 밝혔다."),
    ]


def _fact_sheet(problems=None, **over) -> FactSheet:
    base = dict(
        n_samples=200,
        verdict="FAIL",
        score=50.0,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail", is_core=True),
        ],
        problems=_problems() if problems is None else problems,
    )
    base.update(over)
    return FactSheet(**base)


def _llm_json(recs: list) -> dict:
    return {
        "interpretation": {"confusion_analysis": "", "distribution_analysis": ""},
        "conclusion": {"verdict": "FAIL", "benchmark": "", "narrative": "", "risks": ""},
        "recommendation_narrative": {"data_quality": "", "model_ops": ""},
        "recommendations": recs,
    }


def _rec(pid, priority="LOW", category="개선", action="재학습을 검토한다", impact="재현율 개선"):
    return {"problem_id": pid, "priority": priority, "category": category,
            "action": action, "expected_impact": impact}


async def _run(make_fake_anthropic_client, recs, fs=None):
    fs = fs or _fact_sheet()
    client = make_fake_anthropic_client(_llm_json(recs))
    req = NarrativeRequest(task_type=TaskType.binary, report_purpose="internal", fact_sheet=fs)
    return await generate_narrative(client, req)


# ── 스키마(요청) ────────────────────────────────────────────────────────────────

def test_fact_sheet_problems_default_empty():
    assert FactSheet(verdict="PASS", score=100.0).problems == []


def test_problem_fact_rejects_bad_shapes():
    with pytest.raises(ValidationError):
        _problem("P1", "core_fail", "URGENT")                       # 우선순위는 세 값뿐
    with pytest.raises(ValidationError):
        ProblemFact(id="P1", kind="made_up", priority="LOW", title="x")  # 종류는 9종뿐
    with pytest.raises(ValidationError):
        ProblemFact(id="X1", kind="core_fail", priority="HIGH", title="x")  # 번호 꼴은 P1…
    with pytest.raises(ValidationError):
        _problem("P1", "core_fail", "HIGH", evidence={"blob": "가" * MAX_PROBLEM_EVIDENCE_CHARS})


def test_fact_sheet_problems_bounded_and_unique():
    many = [_problem(f"P{i + 1}", "label_caution", "LOW") for i in range(MAX_PROBLEMS + 1)]
    with pytest.raises(ValidationError):
        _fact_sheet(problems=many)
    _fact_sheet(problems=many[:MAX_PROBLEMS])  # 상한까지는 받는다
    with pytest.raises(ValidationError):
        _fact_sheet(problems=[_problem("P1", "core_fail", "HIGH"), _problem("P1", "label_caution", "LOW")])


# ── LLM 응답 스키마·프롬프트 ────────────────────────────────────────────────────

def test_response_schema_problem_id_is_enum_of_request_ids():
    schema = build_response_schema(_problems())
    recs = schema["properties"]["recommendations"]
    assert recs["items"]["properties"]["problem_id"] == {"type": "string", "enum": ["P1", "P2", "P3"]}
    assert "problem_id" in recs["items"]["required"]
    assert set(recs["items"]["required"]) == set(recs["items"]["properties"])  # strict: 전부 required
    assert "maxItems" not in recs


def test_response_schema_without_problems_has_no_recommendations_property():
    """문제가 없으면 권고 속성 자체가 없다 — 구조화 출력은 maxItems 를 받지 않아 '빈 배열만'을 말할 수 없다."""
    for problems in ([], None):
        schema = build_response_schema(problems)
        assert "recommendations" not in schema["properties"]
        assert "recommendations" not in schema["required"]
        assert set(schema["required"]) == set(schema["properties"])


_FORBIDDEN_SCHEMA_KEYWORDS = {"maxItems", "minimum", "maximum", "minLength", "maxLength"}


def _walk_schema(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk_schema(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_schema(v)


@pytest.mark.parametrize("problems", [None, "some"])
def test_response_schema_uses_only_structured_output_keywords(problems):
    """구조화 출력이 받지 않는 키워드가 없고(minItems 는 0·1 만), 추론·근거 칸이 없다(reasoning_extraction)."""
    schema = build_response_schema(_problems() if problems else None)
    nodes = list(_walk_schema(schema))
    assert len(nodes) > 5
    assert not [n for n in nodes if _FORBIDDEN_SCHEMA_KEYWORDS & set(n)]
    assert all(n.get("minItems", 0) in (0, 1) for n in nodes)
    names = {k.lower() for n in nodes for k in (n.get("properties") or {})}
    assert names and not {k for k in names if any(w in k for w in ("reason", "rationale", "trace", "thought"))}
    for n in nodes:
        if n.get("type") == "object":
            assert n.get("additionalProperties") is False
            assert set(n["required"]) == set(n["properties"])


def test_user_prompt_explains_problems():
    fs = _fact_sheet()
    text = build_user_prompt(fs.model_dump(), [], {})
    assert "problem_id" in text and "problems 의 id 하나" in text
    assert "expected_impact(기대 효과)에는 숫자나 예상 수치" in text
    assert "recommend_section 이 9.1" in text
    empty = build_user_prompt(_fact_sheet(problems=[]).model_dump(), [], {})
    assert "recommendations 는 빈 배열로 두고" in empty
    assert "problems 의 id 하나" not in empty


# ── 서버 후처리(LLM 경로) ───────────────────────────────────────────────────────

async def test_llm_recommendations_drop_unknown_and_empty_ids(make_fake_anthropic_client):
    resp = await _run(make_fake_anthropic_client, [
        _rec("P9", action="목록에 없는 문제"),
        _rec("", action="번호 없는 권고"),
        _rec("   ", action="공백 번호"),
        _rec("P1", action="P1 권고"),
        _rec("P2", action="P2 권고"),
        _rec("P3", action="P3 권고"),
    ])
    assert resp.meta.source == "llm"
    assert [r.action for r in resp.recommendations] == ["P1 권고", "P2 권고", "P3 권고"]
    assert {r.problem_id for r in resp.recommendations} == {"P1", "P2", "P3"}


async def test_llm_recommendations_fill_missing_problem_with_template(make_fake_anthropic_client):
    resp = await _run(make_fake_anthropic_client, [_rec("P2", action="임계값 다시 고르기")])
    assert resp.meta.source == "llm"
    ids = [r.problem_id for r in resp.recommendations]
    assert ids == ["P1", "P2", "P3"]
    p1, p2, p3 = resp.recommendations
    assert (p1.category, p1.action, p1.expected_impact) == _PROBLEM_TEMPLATES["core_fail"]
    assert p2.action == "임계값 다시 고르기"
    assert p3.category == _PROBLEM_TEMPLATES["imbalance_effect"][0]


async def test_llm_recommendations_order_by_problem_then_llm_order(make_fake_anthropic_client):
    resp = await _run(make_fake_anthropic_client, [
        _rec("P3", action="c1"), _rec("P1", action="a1"), _rec("P2", action="b1"),
        _rec("P1", action="a2"), _rec("P3", action="c2"),
    ])
    assert [r.action for r in resp.recommendations] == ["a1", "a2", "b1", "c1", "c2"]


async def test_llm_recommendations_not_truncated_to_five(make_fake_anthropic_client):
    recs = [_rec(pid, action=f"권고 {pid}-{k}") for pid in ("P1", "P2", "P3") for k in "abc"]
    resp = await _run(make_fake_anthropic_client, recs)
    assert len(resp.recommendations) == 9


async def test_llm_priority_forced_from_problem(make_fake_anthropic_client):
    resp = await _run(make_fake_anthropic_client, [
        _rec("P1", priority="LOW"), _rec("P2", priority="HIGH"), _rec("P3", priority="LOW"),
    ])
    assert [r.priority for r in resp.recommendations] == ["HIGH", "MEDIUM", "MEDIUM"]


@pytest.mark.parametrize("impact", ["재현율 0.8 달성", "통과율 50% 회복", "200건 전부 정분류"])
async def test_impact_with_number_falls_back_whole(make_fake_anthropic_client, impact):
    """기대 효과의 숫자는 허용 목록 안(목표값 0.8·점수 50·표본 200)이어도 예측값이라 받지 않는다 — 전체 폴백.

    허용 목록 밖 숫자는 그보다 먼저 grounding 이 잡으므로, 여기서는 grounding 을 통과하는 숫자로 고정한다.
    """
    resp = await _run(make_fake_anthropic_client, [
        _rec("P1"), _rec("P2", impact=impact), _rec("P3"),
    ])
    assert resp.meta.source == "fallback"
    assert resp.meta.reason == "impact_number"
    assert [r.problem_id for r in resp.recommendations] == ["P1", "P2", "P3"]
    assert all(not has_number_token(r.expected_impact) for r in resp.recommendations)


async def test_impact_number_ignored_on_dropped_recommendation(make_fake_anthropic_client):
    """버려지는 권고(모르는 번호)의 기대 효과는 성적서에 실리지 않으니 폴백 사유가 아니다."""
    resp = await _run(make_fake_anthropic_client, [_rec("P7", impact="0.8 달성"), _rec("P1")])
    assert resp.meta.source == "llm"
    assert [r.problem_id for r in resp.recommendations] == ["P1", "P2", "P3"]


def test_has_number_token_follows_grounding_tokens():
    assert not has_number_token("F1 Score 와 P99 지연 개선, 9절 참조")  # 식별자·절 번호는 숫자가 아니다
    assert not has_number_token("")
    assert has_number_token("오류 10% 감소")
    assert has_number_token("-0.1 개선")


async def test_llm_recommendations_dropped_when_no_problems(make_fake_anthropic_client):
    resp = await _run(make_fake_anthropic_client, [_rec("P1")], fs=_fact_sheet(problems=[]))
    assert resp.meta.source == "llm"
    assert resp.recommendations == []


async def test_llm_schema_sent_with_request_problem_ids(make_fake_anthropic_client):
    fs = _fact_sheet()
    client = make_fake_anthropic_client(_llm_json([_rec("P1")]))
    await generate_narrative(client, NarrativeRequest(task_type=TaskType.binary, fact_sheet=fs))
    sent = client.beta.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
    enum = sent["properties"]["recommendations"]["items"]["properties"]["problem_id"]
    assert enum["enum"] == ["P1", "P2", "P3"]


# ── 폴백 ───────────────────────────────────────────────────────────────────────

def test_every_kind_has_template_without_numbers():
    kinds = ProblemFact.model_fields["kind"].annotation.__args__
    assert set(_PROBLEM_TEMPLATES) == set(kinds)
    for kind, (category, action, impact) in _PROBLEM_TEMPLATES.items():
        assert category and action and impact, kind
        for text in (category, action, impact):
            assert not has_number_token(text), (kind, text)


def test_fallback_recommendations_one_template_per_problem():
    fs = _fact_sheet()
    resp = build_fallback_narrative(fs, [], compute_derived(fs), reason="no_key")
    assert [(r.problem_id, r.priority) for r in resp.recommendations] == [
        ("P1", "HIGH"), ("P2", "MEDIUM"), ("P3", "MEDIUM"),
    ]
    assert resp.recommendations[1].category == _PROBLEM_TEMPLATES["threshold_tuning"][0]
    # 불균형 + 편향 노력 "수행 안 함" 메모 → 권고 앞에 그 사실
    assert resp.recommendations[2].action.startswith("의뢰자가 편향 감소 노력을 수행하지 않았다고")
    assert not problem_recommendation(_problem("P1", "imbalance_effect", "MEDIUM")).action.startswith("의뢰자가")


def test_fallback_no_problems_no_recommendations():
    """종전 충돌 4곳(D22): 문제 없는 '운영 모니터링', 제외 행 권고, IR>1.5 권고, 미달 지표 권고가 없다."""
    fs = _fact_sheet(problems=[], dropped_rows=3, distribution={"class_distribution": {"0": 150, "1": 50},
                                                                 "imbalance_ratio": 3.0})
    resp = build_fallback_narrative(fs, [], compute_derived(fs), reason="no_key")
    assert resp.recommendations == []
    narr = resp.recommendation_narrative
    assert narr.data_quality == "데이터 측면에서 권고 대상으로 지목된 문제는 없다. 전처리에서 3건이 평가에서 제외되었다."
    assert narr.model_ops == "모델 측면에서 권고 대상으로 지목된 문제는 없다."
    for text in (narr.data_quality, narr.model_ops):
        assert "권고한다" not in text and "모니터링" not in text and "Recall" not in text


def test_fallback_narrative_splits_problems_by_section():
    fs = _fact_sheet()
    narr = build_fallback_narrative(fs, [], compute_derived(fs), reason="no_key").recommendation_narrative
    assert "P3(불균형 영향 있음 (다수 0 · 소수 1))" in narr.data_quality
    assert "P1" not in narr.data_quality and "P2" not in narr.data_quality
    assert "P1(핵심 지표 미달 (Recall))" in narr.model_ops and "P2(임계값 부적합)" in narr.model_ops
    assert "P3" not in narr.model_ops

    data_only = _fact_sheet(problems=[_problem("P1", "label_caution", "LOW", "정답 라벨 주의")])
    narr = build_fallback_narrative(data_only, [], compute_derived(data_only), "no_key").recommendation_narrative
    assert narr.model_ops == "모델 측면에서 권고 대상으로 지목된 문제는 없다."


def test_fallback_with_problems_passes_grounding():
    fs = _fact_sheet()
    derived = compute_derived(fs)
    refs = build_benchmark_refs("binary", fs.metrics)
    resp = build_fallback_narrative(fs, refs, derived, reason="no_key")
    texts = _collect_grounding_texts(resp.model_dump(exclude={"meta", "benchmark_refs"}))
    g = verify_grounding(texts, build_number_whitelist(fs, refs, derived))
    assert g.passed, g.violations


# ── grounding 허용 목록 ─────────────────────────────────────────────────────────

def test_whitelist_takes_problem_evidence_recursively():
    without = build_number_whitelist(_fact_sheet(problems=[]), [], {})
    for tok in ("0.1409", "0.37", "3.2", "0.8421"):
        assert tok not in without, tok
    wl = build_number_whitelist(_fact_sheet(), [], {})
    for tok in ("0.1409", "14.09", "0.37", "3.2", "0.8421", "0.7012"):
        assert tok in wl, tok
    # 중첩 dict·문자열 속 숫자(이름 속 숫자 꼴 클래스)도
    nested = _fact_sheet(problems=[_problem(
        "P1", "distribution_shift", "LOW", "학습↔평가 분포 차이 (7)",
        {"7": {"train": 0.412, "eval": "0.288"}}, note="표본 4321건")])
    wl = build_number_whitelist(nested, [], {})
    for tok in ("0.412", "0.288", "7", "4321"):
        assert tok in wl, tok
