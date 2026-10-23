"""test_s6_templates.py — 규칙 템플릿 다시 쓰기(서술 라운드 S6 — 세션 A): 결론 목록을 읽는 칸 템플릿.

완료 기준(계획 "세션 재구성" A·S6 검증)을 장치마다 따로 시험한다.
- 작업 유형 × 판정 4가지 — 템플릿 출력이 자기 검문(숫자 허용 목록·판정 모순·영어 코드·D24 선언 숫자)을 통과한다.
- 고정 문구 17·18 을 성적서가 찍으면 그 숫자 묶음이 서술에 다시 나오지 않는다(옛 요청은 템플릿이 한 번 쓴다).
- 옛 경로 골든(옛 프론트 모양 사실 시트)과 새 골든(작업 경로·계약 예시)을 따로 둔다.
- 결론 목록을 읽는다(빠졌던 항목), 고객 입력(클래스 설명 D37), facts·task_type 없는 옛 요청, 조사·날값·개조식,
  해석형 항목은 사실 문장만, 신뢰구간 두 갈래·생략 사유, 17·18 해석 재료(우세한 오류·근소 지표의 판정).
바탕값은 다른 칸에서 우연히 같은 숫자가 나오지 않게 고른다(판정을 바꿀 때 목표값 차이 0.0731 — 라운드 작업 함정).
"""
import copy
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.narrative import checks, fallback, generator, prompt
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.grounding import build_number_whitelist, find_verdict_contradictions, verify_grounding
from app.narrative.required_items import code_prints, error_ratio_sentence, select_items
from app.narrative.schemas import ConfusionCountFact, ErrorRatiosFact, FactSheet, MetricFact, NarrativeRequest
from app.narrative.sections import SECTION_IDS
from app.narrative.wording import VERDICT_LABEL, josa
from golden_utils import assert_golden

CONTRACT = Path(__file__).parent / "contract"
TASKS = ("binary", "multiclass", "multilabel")
VERDICTS = ("PASS", "CONDITIONAL_PASS", "FAIL", "INCONCLUSIVE")
DESCRIPTIONS = {
    "binary": {"0": "정상", "1": "불량"},
    "multiclass": {"cat": "고양이", "dog": "개", "bird": "새"},
    "multilabel": {"sports": "스포츠 기사", "tech": "기술 기사"},
}
PURPOSE = "생산 라인 2 의 불량 검출"
# 판정을 바꿀 때 목표값을 값에서 이만큼 떨어뜨린다 — 다른 사실에서 나오지 않는 수(우연히 허용되는 숫자가 없게).
GAP = 0.0731


def _raw(name: str) -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text())


def _client_inputs(task: str) -> dict:
    return {"class_label_descriptions": DESCRIPTIONS[task], "model": {"모델 용도": PURPOSE}}


def _judge(m: dict, passed: bool) -> None:
    """지표 하나를 충족/미달로 — 목표값을 값에서 GAP 만큼 떨어뜨리고 판정 칸(마진·근소·구간 포함)을 맞춘다."""
    v = m["value"]
    higher = m.get("higher_is_better") is not False
    t = round(v - GAP if passed == higher else v + GAP, 4)
    margin = round((v - t) if higher else (t - v), 9)
    m.update(threshold=t, status="pass" if passed else "fail", margin=margin,
             margin_text=f"{round(margin, 4):g}", near_target=False)
    if m.get("ci_low") is not None and m.get("ci_high") is not None:
        m["ci_contains_target"] = m["ci_low"] <= t <= m["ci_high"]


def _ref(m: dict) -> dict:
    return {"metric_id": m["metric_id"], "display_name": m["display_name"]}


def _problem(pid: str, kind: str, priority: str, title: str, evidence: dict, section: str = "9.2") -> dict:
    return {"id": pid, "kind": kind, "priority": priority, "title": title, "source_section": "8.2",
            "recommend_section": section, "evidence": evidence, "note": None}


def _request(task: str, verdict: str, *, client_inputs: bool = True, latency: bool = True) -> NarrativeRequest:
    """계약 예시(최종 불합격)를 판정 하나로 바꾼다. FAIL 은 예시 그대로(근소 차이·고정 문구 18 포함)."""
    raw = _raw(task)
    fs = raw["fact_sheet"]
    facts = fs["facts"]
    if latency:
        fs["latency"] = {"available": True, "mean": 12.5, "p50": 10.25, "p95": 30.75, "p99": 41.5, "unit": "ms"}
    if verdict != "FAIL":
        judged = [m for m in fs["metrics"] if m.get("threshold") is not None]
        kept = [p for p in fs["problems"] if p["kind"] not in ("core_fail", "noncore_fail", "core_near_target", "inconclusive")]
        new: list[dict] = []
        vd = {"verdict": verdict, "failed_core": [], "failed_noncore": [], "unavailable_core": [],
              "inconclusive_cause": None, "inconclusive_metrics": []}
        if verdict == "PASS":
            for m in judged:
                _judge(m, True)
        elif verdict == "CONDITIONAL_PASS":
            noncore = next((m for m in reversed(judged) if not m["is_core"]), judged[-1])
            for m in judged:
                m["is_core"] = m["is_core"] and m is not noncore
                _judge(m, m is not noncore)
            vd["failed_noncore"] = [_ref(noncore)]
            new.append(_problem("P1", "noncore_fail", "MEDIUM", f"비핵심 지표 미달 ({noncore['display_name']})",
                                {noncore["display_name"]: {"value": noncore["value"], "target": noncore["threshold"]}}))
        else:  # INCONCLUSIVE — 핵심 지표 하나가 산출되지 않았다(지표 목록에서 빠지고 판정 사유에 남는다)
            core = next(m for m in judged if m["is_core"])
            fs["metrics"] = [m for m in fs["metrics"] if m is not core]
            judged = [m for m in judged if m is not core]
            for m in judged:
                _judge(m, True)
            vd.update(unavailable_core=[_ref(core)], inconclusive_cause="core_not_computed", inconclusive_metrics=[_ref(core)])
            new.append(_problem("P1", "inconclusive", "HIGH", f"판정 불가 원인 ({core['display_name']})",
                                {"cause": "핵심 지표 산출 실패", "metrics": [core["display_name"]]}))
        problems = new + kept
        for i, p in enumerate(problems, 1):
            p["id"] = f"P{i}"
        fs["problems"] = problems
        passed = sum(1 for m in judged if m["status"] == "pass")
        vd.update(judged_count=len(judged), passed_count=passed)
        facts["verdict_detail"] = vd
        fs["verdict"] = verdict
        fs["score"] = round(passed / len(judged) * 100, 1) if judged else 0.0
        sentences = [s for s in facts["fixed_sentences"] if s["id"] not in ("18", "6")]
        if not problems:
            sentences.append({"id": "6", "section": "9", "text": "지목된 문제가 없어 별도의 개선 권고는 없다."})
        facts["fixed_sentences"] = sentences
    if client_inputs:
        raw["client_inputs"] = _client_inputs(task)
    return NarrativeRequest.model_validate(raw)


def _materials(req: NarrativeRequest):
    fs, task, ci = req.fact_sheet, req.task_type.value, req.client_inputs
    derived = compute_derived(fs, task_type=task)
    refs = build_benchmark_refs(task, fs.metrics)
    return fs, task, ci, derived, refs


def _sections(req: NarrativeRequest) -> dict[str, dict]:
    fs, task, ci, derived, refs = _materials(req)
    cons = select_items(fs, ci, task)
    return {
        sid: generator._rec_dicts(fallback.section_template(
            sid, fs, refs, derived, task_type=task, client_inputs=ci, conclusions=cons,
        ))
        for sid in SECTION_IDS
    }


def _all_texts(parts_by_section: dict[str, dict]) -> list[str]:
    return [t for sid, parts in parts_by_section.items() for t in checks.section_texts(sid, parts) if t]


# ── 완료 기준 1: 작업 유형 × 판정 4가지 — 자기 검문 ──────────────────────────────────


@pytest.mark.parametrize("verdict", VERDICTS)
@pytest.mark.parametrize("task", TASKS)
def test_templates_pass_their_own_checks(task, verdict):
    req = _request(task, verdict)
    fs, task, ci, derived, refs = _materials(req)
    assert fs.verdict == verdict
    whitelist = build_number_whitelist(fs, refs, derived, ci)
    conclusions = select_items(fs, ci, task)
    exempt = checks.name_exemptions(fs, ci)
    problem_ids = [p.id for p in fs.problems]
    by_section = _sections(req)
    for sid, parts in by_section.items():
        texts = checks.section_texts(sid, parts)
        if sid == "9" and not fs.problems:
            # 문제가 0개면 9칸은 AI 를 부르지 않고 검사하지 않는다(출처 fixed) — 숫자만 본다.
            assert verify_grounding(texts, whitelist).passed, texts
            continue
        result = checks.check_section(
            sid, parts, whitelist=whitelist, verdict=fs.verdict, conclusions=conclusions,
            problem_ids=problem_ids, exempt=exempt,
        )
        assert result.hard_failures == [], (sid, result.as_record(), texts)
        assert not result.d24_missing, (sid, texts)
    texts = _all_texts(by_section)
    assert find_verdict_contradictions(texts, fs.verdict) == []
    # 이빨 — 검사가 실제로 숫자를 봤고, 판정 문장이 이 판정 이름을 쓴다(판정을 바꾼 픽스처가 그 판정이다).
    assert verify_grounding(texts, whitelist).checked > 20
    assert f"'{VERDICT_LABEL[verdict]}'" in by_section["8.2"]["conclusion.narrative"]


def test_self_check_has_teeth():
    """자기 검문이 헛돌지 않는다 — 템플릿 글의 숫자 하나를 바꾸면 숫자 검사, 판정을 뒤집으면 판정 모순에 걸린다."""
    req = _request("binary", "FAIL")
    fs, task, ci, derived, refs = _materials(req)
    whitelist = build_number_whitelist(fs, refs, derived, ci)
    parts = _sections(req)["7.1"]
    text = parts["interpretation.confusion_analysis"]
    assert "99건" in text
    broken = {"interpretation.confusion_analysis": text.replace("99건", "97건")}
    result = checks.check_section("7.1", broken, whitelist=whitelist, verdict="FAIL", conclusions=[],
                                  problem_ids=[], exempt=set())
    assert "numbers" in result.hard_failures
    pass_req = _request("binary", "PASS")
    narrative = _sections(pass_req)["8.2"]["conclusion.narrative"]
    assert find_verdict_contradictions([narrative], "FAIL")  # 합격 문장은 불합격 판정과 모순


# ── 완료 기준 2: 17·18 숫자 묶음이 두 번 나오지 않는다 ─────────────────────────────────

# 계약 예시의 고정 문구 17 숫자(확신·경계 오류의 건수·비율) — 7.1 결론(오분류 총건수)과 겹치는 분모는 뺀다.
SEVENTEEN = {
    "binary": ("31건", "31.3%"),
    "multiclass": ("27건", "20.3%", "1.5%"),
    "multilabel": ("46", "11.1%"),
}


@pytest.mark.parametrize("task", TASKS)
def test_fixed_17_numbers_are_not_repeated_when_the_report_prints_them(task):
    req = _request(task, "FAIL")
    assert code_prints(req.fact_sheet, "17")
    texts = _all_texts(_sections(req))
    for token in SEVENTEEN[task]:
        assert not any(token in t for t in texts), (token, texts)
    # 해석 한 문장은 있다(7.1 — 우세한 오류 또는 확신 오류의 뜻).
    assert "확신 오류" in _sections(req)["7.1"]["interpretation.confusion_analysis"]


def test_fixed_18_numbers_are_not_repeated_and_the_interpretation_follows_the_verdicts():
    req = _request("binary", "FAIL")  # 근소 지표: Accuracy(미달)·F1 Score·F-beta Score(충족)
    assert code_prints(req.fact_sheet, "18")
    risks = _sections(req)["8.3"]["conclusion.risks"]
    for token in ("0.512", "0.0123", "3%", "아슬아슬"):
        assert token not in risks, token
    assert "충족과 미달이 서로 바뀔 수 있어" in risks


def test_old_requests_get_the_17_18_numbers_once():
    """옛 프론트(facts 없음)는 17·18 을 찍지 않는다 — 템플릿이 숫자를 한 번 쓴다."""
    req = NarrativeRequest.model_validate(_raw("legacy_main"))
    assert not code_prints(req.fact_sheet, "17") and not code_prints(req.fact_sheet, "18")
    texts = _all_texts(_sections(req))
    joined = " ".join(texts)
    assert joined.count("31건(31.3%)") == 1
    assert joined.count("F1 Score(0.512, 목표 0.5, 차이 0.012)") == 1


def test_trimmed_fixed_sentences_still_count_as_printed():
    """facts 상한으로 문장 목록을 비웠어도 화면은 17·18 을 찍는다(프론트 errorRatioSentence·nearTargetNote 는 자르기와 무관)."""
    raw = _raw("binary")
    raw["fact_sheet"]["facts"]["fixed_sentences"] = []
    raw["fact_sheet"]["facts"]["omitted"] = {"fixed_sentences": 3}
    req = NarrativeRequest.model_validate(raw)
    assert code_prints(req.fact_sheet, "17")
    by_section = _sections(req)
    assert "31건" not in " ".join(_all_texts(by_section))
    assert "아슬아슬" not in by_section["8.3"]["conclusion.risks"]  # 9절 문제 이름('핵심 지표 아슬아슬 합격')은 프론트 문안


# ── 완료 기준 3: 옛 경로 골든과 새 골든 ─────────────────────────────────────────────


@pytest.fixture
def _no_client():
    prev = getattr(app.state, "anthropic_client", None)
    app.state.anthropic_client = None
    try:
        yield
    finally:
        app.state.anthropic_client = prev


def test_legacy_route_golden_with_production_front_shape(_no_client):
    """옛 경로(/api/generate-narrative)·옛 프론트 모양 사실 시트 — 규칙 문장이 이유 없이 바뀌지 않게."""
    body = TestClient(app).post("/api/generate-narrative", json=_raw("legacy_main"))
    assert body.status_code == 200, body.text
    data = body.json()
    assert data["meta"]["source"] == "fallback"
    assert_golden("narrative_legacy_main_fallback", data)


@pytest.mark.parametrize("task", TASKS)
async def test_job_path_golden_with_contract_examples(task):
    """새 경로(작업 생성기, 키 없음 → 칸마다 규칙 문장) — 계약 예시 + 클래스 설명·모델 용도."""
    req = _request(task, "FAIL", latency=False)
    result = await generator.generate_sections(None, req)
    assert result.response.meta.reason == "no_key"
    assert_golden(f"narrative_job_fallback_{task}", {
        "response": result.response.model_dump(mode="json"),
        "section_sources": result.section_sources,
        "section_models": result.section_models,  # 키 없음 — 칸마다 None(사용자 결정 2026-10-22)
    })


# ── 결론 목록을 읽는다 — 빠졌던 항목 ────────────────────────────────────────────────


def test_items_missing_from_the_old_templates_are_written():
    binary = _sections(_request("binary", "FAIL"))
    c71 = binary["7.1"]["interpretation.confusion_analysis"]
    assert "의뢰자가 더 위험하다고 밝힌 거짓 음성(FN, 놓침)은 43건으로 전체 오류 99건의 43.4%이다." in c71  # 선언 위험(D24)
    assert "F1 이 가장 낮은 클래스는 F1 0.4975인 0(정상)이며" in c71  # 최약 클래스
    risks = binary["8.3"]["conclusion.risks"]
    assert "지연시간은 평균 12.5ms, 중앙값(p50) 10.25ms, 95번째 백분위수(p95) 30.75ms, 99번째 백분위수(p99) 41.5ms이다." in risks
    assert "대화형(실시간) 사용을 밝혔으므로" in risks  # D24 8.3 — 숫자는 같은 칸의 지연시간 문장
    multiclass = _sections(_request("multiclass", "FAIL"))
    c71 = multiclass["7.1"]["interpretation.confusion_analysis"]
    assert "가장 많이 헷갈린 쌍(실제→예측)은 dog(개)→cat(고양이) 28건(전체 오분류의 21.1%)" in c71  # 최다 오분류 쌍
    assert "학습 분포와 평가 분포의 비율 차가 10%p 이상인 클래스는" in multiclass["7.2"]["interpretation.distribution_analysis"]
    assert "가중 평균 F1 은 0.3363이다" in multiclass["8.2"]["conclusion.narrative"]  # 선언한 평균 방식(D24 8.2)


def test_interpretive_items_are_fact_sentences_only():
    cons = {c.item_id: c for c in select_items(*(lambda r: (r.fact_sheet, r.client_inputs, r.task_type.value))(_request("binary", "FAIL")))}
    assert cons["8.2-13"].template_text == f"의뢰자가 밝힌 모델 용도는 '{PURPOSE}'이다."
    assert cons["8.3-01"].template_text == "F1 이 가장 낮은 클래스는 F1 0.4975인 0(정상)이다."
    assert "해석" in cons["8.2-13"].text and "위험을 쓴다" in cons["8.3-01"].text  # AI 결론은 해석을 맡긴다


_AI_ONLY = ("고정 문구", "되풀이", "결론 목록", "사실 시트", "성적서가 따로", "해석한다", "쓴다.", "찍는다")


@pytest.mark.parametrize("task, verdict", [(t, v) for t in TASKS for v in VERDICTS] + [("legacy_main", "FAIL")])
def test_template_texts_follow_writing_rules(task, verdict):
    """작성 규칙(D37) — 서술체(개조식 ';'·줄바꿈·글머리·괄호 밖 '항목: 값' 없음), AI 용 설명이 새지 않음."""
    if task == "legacy_main":
        req = NarrativeRequest.model_validate(_raw("legacy_main"))
    else:
        req = _request(task, verdict)
    for text in _all_texts(_sections(req)):
        assert ";" not in text and "\n" not in text and not text.lstrip().startswith(("-", "•", "*")), text
        outside = re.sub(r"\([^()]*(?:\([^()]*\)[^()]*)*\)", "", text)
        assert ":" not in outside, text
        for word in _AI_ONLY:
            assert word not in text, (word, text)


# ── 고객 입력(D37 클래스 설명)·배선 ───────────────────────────────────────────────


def test_class_names_carry_client_descriptions():
    c71 = _sections(_request("binary", "FAIL"))["7.1"]["interpretation.confusion_analysis"]
    assert "양성 클래스 1(불량) 기준으로 거짓 양성(FP)은 56건" in c71
    assert "양성 클래스 1(불량)로 예측한 것은 108건" in c71  # 조사 — '1' 은 ㄹ 받침이라 '로'
    c72 = _sections(_request("binary", "FAIL"))["7.2"]["interpretation.distribution_analysis"]
    assert "0(정상) 105건(52.5%), 1(불량) 95건(47.5%)" in c72
    plain = _sections(_request("binary", "FAIL", client_inputs=False))["7.1"]["interpretation.confusion_analysis"]
    assert "(불량)" not in plain and "양성 클래스 1 기준으로" in plain


def _binary_without_binary_signals() -> NarrativeRequest:
    """이진인데 사실 시트로는 이진임을 짐작할 수 없는 요청(양성 클래스·엔진 줄·행렬 없음) — 작업 유형 배선을 따로 본다."""
    raw = _raw("binary")
    fs = raw["fact_sheet"]
    for key in ("confusion", "f1_threshold_gain"):
        fs[key] = None
    fs["confusion_counts"] = []
    fs["facts"]["positive_class"] = None
    raw["client_inputs"] = _client_inputs("binary")
    return NarrativeRequest.model_validate(raw)


async def test_job_generator_passes_task_type_and_client_inputs_to_templates():
    req = _binary_without_binary_signals()
    result = await generator.generate_sections(None, req)
    narrative = result.response.conclusion.narrative
    assert "양성 클래스의 Precision 은" in narrative  # 이진 전용 항목(8.2-05) — 작업 유형을 넘겨야 나온다
    assert "0(정상)" in result.response.interpretation.confusion_analysis  # 클래스 설명 — 고객 입력을 넘겨야 나온다


def test_legacy_route_passes_task_type_and_client_inputs_to_templates(_no_client):
    req = _binary_without_binary_signals()
    body = TestClient(app).post("/api/generate-narrative", json=json.loads(req.model_dump_json())).json()
    assert "양성 클래스의 Precision 은" in body["conclusion"]["narrative"]
    assert "0(정상)" in body["interpretation"]["confusion_analysis"]


# ── facts·task_type 이 없는 옛 요청 ──────────────────────────────────────────────


def test_old_requests_without_facts_or_task_type_do_not_break():
    req = NarrativeRequest.model_validate(_raw("legacy_main"))
    fs = req.fact_sheet
    derived = compute_derived(fs)  # 작업 유형 없이(옛 호출)
    refs = build_benchmark_refs("binary", fs.metrics)
    resp = fallback.build_fallback_narrative(fs, refs, derived, "no_key")
    texts = [resp.interpretation.confusion_analysis, resp.interpretation.distribution_analysis, resp.conclusion.benchmark,
             resp.conclusion.narrative, resp.conclusion.risks, resp.recommendation_narrative.data_quality,
             resp.recommendation_narrative.model_ops]
    assert all(t.strip() for t in texts)
    assert verify_grounding(texts, build_number_whitelist(fs, refs, derived)).passed
    assert find_verdict_contradictions(texts, fs.verdict) == []
    bare = fallback.build_fallback_narrative(FactSheet(verdict="PASS", score=0.0), [], {}, "no_key")
    assert bare.interpretation.confusion_analysis == "혼동 행렬 데이터가 제공되지 않아 클래스 간 오분류 분석을 생략한다."
    assert bare.interpretation.distribution_analysis == "클래스 분포 데이터가 제공되지 않아 분포 분석을 생략한다."


def _row(label, tp, fp, fn, tn):
    return ConfusionCountFact(label=label, tp=tp, fp=fp, fn=fn, tn=tn)


@pytest.mark.parametrize("task, rows, expected", [
    ("binary", [_row("1", 40, 7, 9, 44)], "혼동행렬 기준 전체 100건 중 16건(16%)이 오분류되었다."),
    ("multiclass", [_row("a", 30, 4, 6, 0), _row("b", 25, 5, 3, 0), _row("c", 20, 3, 3, 0)],
     "혼동행렬 기준 전체 87건 중 12건(13.8%)이 오분류되었다."),
    ("multilabel", [_row("x", 30, 4, 6, 60), _row("y", 25, 5, 3, 67)],
     "레이블별 혼동행렬을 합친 레이블 칸 200칸 중 18칸(9%)이 틀렸다."),
])
def test_engine_counts_give_totals_when_the_matrix_is_missing(task, rows, expected):
    """M21(혼동행렬)을 고르지 않은 옛 요청 — 엔진 클래스별 건수(confusion_counts)로 오분류 총건수를 쓴다."""
    fs = FactSheet(verdict="FAIL", score=50.0, n_samples=87, confusion_counts=rows)
    derived = compute_derived(fs, task_type=task)
    text = fallback.build_fallback_narrative(fs, [], derived, "no_key", task_type=task).interpretation.confusion_analysis
    assert text.startswith(expected), text
    assert verify_grounding([text], build_number_whitelist(fs, [], derived)).passed


def test_engine_counts_without_task_type_are_not_guessed_for_many_classes():
    rows = [_row("a", 30, 4, 6, 0), _row("b", 25, 5, 3, 0)]
    assert "confusion" not in compute_derived(FactSheet(verdict="FAIL", score=0.0, confusion_counts=rows))


# ── 17·18 해석 재료와 해석 한 문장 ─────────────────────────────────────────────────


def _er(boundary, confident, misclassified=40):
    return ErrorRatiosFact(
        unit="sample", misclassified=misclassified,
        boundary=None if boundary is None else {"count": boundary, "ratio": boundary / misclassified if misclassified else None},
        boundary_note="threshold_unknown" if boundary is None else None,
        confident={"count": confident, "ratio": confident / misclassified if misclassified else None},
    )


@pytest.mark.parametrize("boundary, confident, material, sentence", [
    (17, 5, "경계 오류가 확신 오류보다 많다", "경계 오류가 확신 오류보다 많다. 경계 오류는 결정 경계 가까이에서 갈린 오류로"),
    (3, 11, "확신 오류가 경계 오류보다 많다", "확신 오류가 경계 오류보다 많다. 확신 오류는 모델이 틀린 쪽에"),
    (6, 6, "같은 수다", "어느 한쪽 성격의 오류가 우세하지 않다"),
    (0, 0, "모두 없다", "모두 없어"),
    (None, 9, "확신 오류가 있다", "임계값을 조정해도 잘 줄지 않으므로"),
    (None, 0, "확신 오류는 없다", "관찰되지 않았다"),
])
def test_error_nature_material_and_interpretation(boundary, confident, material, sentence):
    raw = _raw("binary")
    raw["fact_sheet"]["error_ratios"] = json.loads(_er(boundary, confident).model_dump_json())
    req = NarrativeRequest.model_validate(raw)
    item = next(c for c in select_items(req.fact_sheet, None, "binary") if c.item_id == "7.1-11")
    assert material in item.text and sentence in item.template_text
    # 재료의 숫자는 '근거 건수' 한 문장에만 있다(세션 C — 문장에 쓰지 말라는 말과 함께). 그 밖에는 숫자가 없다.
    outside = re.sub(r"근거 건수\([^)]*\): [^.]*\.", "", item.text)
    assert not re.search(r"\d", item.template_text) and not re.search(r"\d", outside)


def test_error_nature_without_any_misclassification_has_no_interpretation():
    fs = FactSheet(verdict="PASS", score=100.0, error_ratios=_er(0, 0, misclassified=0))
    item = next(c for c in select_items(fs, None, "binary") if c.item_id == "7.1-11")
    assert item.template_text == "확률이 있는 오분류가 없어 경계 오류·확신 오류 비율은 산출하지 않았다."  # 옛 요청(숫자 문장)


def _near_fs(statuses: tuple[str, ...], *, printed: bool) -> FactSheet:
    metrics = [
        MetricFact(metric_id=f"M{i}", display_name=name, value=0.8017 if s == "pass" else 0.7983, threshold=0.8,
                   status=s, near_target=True, margin=0.0017 if s == "pass" else -0.0017)
        for i, (name, s) in enumerate(zip(("Accuracy", "Recall"), statuses), 1)
    ]
    facts = {"fixed_sentences": [{"id": "18", "section": "8.3", "text": "근소 차이 문장"}]} if printed else None
    verdict = "PASS" if all(s == "pass" for s in statuses) else "FAIL"
    return FactSheet(verdict=verdict, score=50.0, metrics=metrics, facts=facts)


@pytest.mark.parametrize("statuses, material, sentence", [
    (("pass", "pass"), "목표를 충족한 지표는 Accuracy, Recall이다", "충족한 지표는 표본이 달라지면 미달로 바뀔 수 있어"),
    (("fail", "fail"), "목표에 미달한 지표는 Accuracy, Recall이다", "미달한 지표는 표본이 달라지면 충족으로 바뀔 수도 있어"),
    (("pass", "fail"), "충족한 지표는 Accuracy이고, 목표에 미달한 지표는 Recall이다", "충족과 미달이 서로 바뀔 수 있어"),
])
def test_near_target_material_and_interpretation(statuses, material, sentence):
    fs = _near_fs(statuses, printed=True)
    item = next(c for c in select_items(fs, None, "binary") if c.item_id == "8.3-02")
    assert material in item.text and sentence in item.template_text
    assert not re.search(r"\d", item.template_text)
    assert find_verdict_contradictions([item.template_text], fs.verdict) == []


def test_near_target_for_old_requests_lists_passing_metrics_with_numbers():
    item = next(c for c in select_items(_near_fs(("pass", "fail"), printed=False), None, "binary") if c.item_id == "8.3-02")
    assert item.template_text.startswith("목표값과의 차이가 목표값의 3% 미만인 아슬아슬 합격 지표는 Accuracy(0.802, 목표 0.8, 차이 0.002)로,")
    assert "Recall" not in item.template_text


def test_prompt_carries_the_interpretation_material():
    for task, material in (
        ("multiclass", "해석 재료: 경계 오류가 확신 오류보다 많다"),
        ("binary", "해석 재료: 근소 차이 지표 가운데 목표를 충족한 지표는 F1 Score, F-beta Score이고, 목표에 미달한 지표는 Accuracy이다."),
    ):
        req = NarrativeRequest.model_validate(_raw(task))
        fs, task_value, ci, derived, refs = _materials(req)
        ctx = prompt.build_context(fs, derived, refs, task_type=task_value, report_purpose="external",
                                   client_inputs=ci, conclusions=select_items(fs, ci, task_value))
        msg = prompt.build_user_message(ctx, ("7.1",) if task == "multiclass" else ("8.3",))
        assert material in msg


# ── 신뢰구간 두 갈래·생략 사유 ─────────────────────────────────────────────────────


def _ci_fs(**facts) -> FactSheet:
    m = MetricFact(metric_id="M1", display_name="Accuracy", value=0.9, threshold=0.8, status="pass",
                   ci_low=0.86, ci_high=0.94, ci_contains_target=False, value_text="0.9", margin=0.1, margin_text="0.1")
    r = MetricFact(metric_id="M3", display_name="Recall", value=0.85, threshold=0.8, status="pass", value_text="0.85",
                   margin=0.05, margin_text="0.05")
    return FactSheet(verdict="PASS", score=100.0, metrics=[m, r], facts=facts or {})


def _ci_text(fs: FactSheet) -> str:
    return next(c.template_text for c in select_items(fs, None, "binary") if c.item_id == "8.2-03")


def test_ci_sentence_branches():
    partial = _ci_text(_ci_fs(ci_available=True, ci_skipped=[{"metric_id": "M3", "display_name": "Recall", "reason": "점추정 불일치"}]))
    assert partial == (
        "신뢰구간을 산출한 지표(Accuracy)는 95% 신뢰구간이 목표값과 떨어져 있다. "
        "신뢰구간을 산출하지 않은 판정 대상 지표는 Recall(점추정 불일치)이다."
    )
    both = _ci_fs()
    both.metrics[1] = both.metrics[1].model_copy(update={"ci_low": 0.81, "ci_high": 0.89, "ci_contains_target": False})
    assert _ci_text(both).startswith("판정 대상 지표의 95% 신뢰구간이 모두 목표값과 떨어져 있어")
    none = FactSheet(verdict="PASS", score=100.0, metrics=[_ci_fs().metrics[1]], facts={"ci_available": False})
    assert _ci_text(none) == "이 평가에서는 판정 대상 지표의 신뢰구간을 산출하지 않았다."
    old = FactSheet(verdict="PASS", score=100.0, metrics=[_ci_fs().metrics[1]])
    assert all(c.item_id != "8.2-03" or c.template_text is None for c in select_items(old, None, "binary"))


def test_ci_end_gets_more_digits_when_it_would_look_like_the_target():
    m = MetricFact(metric_id="M1", display_name="Accuracy", value=0.83, threshold=0.8, status="pass", ci_low=0.80004,
                   ci_high=0.86, ci_contains_target=False)
    other = MetricFact(metric_id="M3", display_name="Recall", value=0.75, threshold=0.7, status="pass", ci_low=0.69,
                       ci_high=0.81, ci_contains_target=True)
    text = _ci_text(FactSheet(verdict="PASS", score=100.0, metrics=[other, m]))
    assert "그 밖에 신뢰구간을 산출한 지표(Accuracy)" in text
    near = FactSheet(verdict="PASS", score=100.0, metrics=[m.model_copy(update={"ci_contains_target": True, "ci_low": 0.79996})])
    assert "구간 0.79996~0.86, 목표 0.8" in _ci_text(near)  # 셋째·넷째 자리로는 0.8 이라 목표값에 닿은 듯 보인다


# ── 조사·날값 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("word, pair, expected", [
    ("0.03", "으로/로", "으로"), ("0.37", "으로/로", "로"), ("0.5", "으로/로", "로"), ("10", "으로/로", "으로"),
    ("0.1377", "은/는", "은"), ("0.5", "은/는", "는"), ("0.136", "은/는", "은"), ("12%", "은/는", "는"),
    ("1(불량)", "으로/로", "로"), ("0(정상)", "으로/로", "으로"), ("cat(고양이)", "으로/로", "으로"),
    ("레이블", "은/는", "은"), ("클래스", "은/는", "는"), ("Recall", "은/는", "은"), ("Precision", "으로/로", "으로"),
    ("F1 Score", "은/는", "는"), ("MCC", "이/가", "가"), ("URL", "이/가", "이"), ("spam", "으로/로", "으로"),
    ("'모델 용도'", "이/가", "가"),
])
def test_josa(word, pair, expected):
    assert josa(word, pair) == expected


def test_benchmark_values_are_not_raw():
    """날값 — 판정 대상이 아닌 지표는 넷째 자리(0.48148148148148145 를 그대로 찍지 않는다)."""
    benchmark = _sections(NarrativeRequest.model_validate(_raw("legacy_main")))["8.1"]["conclusion.benchmark"]
    assert "Precision(0.4815, 참조 범위 0.8~0.92)" in benchmark
    assert not re.search(r"\d\.\d{5,}", benchmark)


def test_error_ratio_sentence_uses_cells_for_multilabel():
    er = ErrorRatiosFact(unit="label_cell", misclassified=12, boundary={"count": 5, "ratio": 0.4167},
                         confident={"count": 2, "ratio": 0.1667})
    assert error_ratio_sentence(er).startswith("틀린 레이블 칸 12칸 중 결정 경계와의 거리가 0.1 미만인 경계 오류는 5칸(41.7%)")


def test_copied_requests_are_not_mutated_by_templates():
    req = _request("multiclass", "FAIL")
    before = copy.deepcopy(req.model_dump())
    _sections(req)
    assert req.model_dump() == before


# ── 변이 보강(세션 A) ────────────────────────────────────────────────────────


@pytest.mark.parametrize("other, expected", [(0.3363, "가중 평균 F1 이 매크로 F1 보다 0.0049 높다"),
                                              (0.3265, "가중 평균 F1 이 매크로 F1 보다 0.0049 낮다")])
def test_average_gap_direction_follows_the_sign(other, expected):
    """8.2 macro vs weighted — difference = 다른 평균 − 매크로(프론트 averageGap). 방향 낱말이 부호를 따른다."""
    diff = round(other - 0.3314, 4)
    facts = {"selected_metrics": ["M1", "M22"], "average_gap": {
        "kind": "macro_vs_weighted", "macro_f1": 0.3314, "other_f1": other, "difference": diff,
        "difference_text": f"{diff:g}", "reference_class": {"label": "dog", "support": 74, "f1": 0.41}}}
    fs = FactSheet(verdict="FAIL", score=0.0, facts=facts)
    text = next(c.template_text for c in select_items(fs, {"class_label_descriptions": {"dog": "개"}}, "multiclass")
                if c.item_id == "8.2-07")
    assert expected in text and "표본이 가장 많은 클래스는 dog(개)이며 표본 74건, F1 0.41이다." in text
    assert verify_grounding([text], build_number_whitelist(fs, [], compute_derived(fs, task_type="multiclass"))).passed


@pytest.mark.parametrize("word, pair, expected", [
    ("0.1", "은/는", "은"), ("21", "이/가", "이"), ("0.7", "과/와", "과"), ("0.8", "을/를", "을"), ("0.2", "은/는", "는"),
])
def test_josa_after_digits_with_rieul_final(word, pair, expected):
    """1(일)·7(칠)·8(팔)은 받침이 있다 — '으로/로' 만 보면 ㄹ 받침과 받침 없음이 같은 답이라 다른 조사로도 본다."""
    assert josa(word, pair) == expected


# ── 독립 검토 반영(세션 A, wf_48ea9448-db9) ─────────────────────────────────────


def test_trimmed_class_table_does_not_claim_a_ranking_over_all_classes():
    """P1 — 클래스가 많아 프론트가 표를 표본 수 순으로 잘랐으면 '오류가 가장 많은'·'클래스별 오류는'을 단정하지 않는다."""
    raw = _raw("multiclass")
    raw["fact_sheet"]["facts"]["omitted"] = {"confusion_counts": 744, "per_class": 744, "class_distribution": 744}
    text = _sections(NarrativeRequest.model_validate(raw))["7.1"]["interpretation.confusion_analysis"]
    assert "클래스가 많아 표에는 일부 클래스만 실렸으며, 실린 클래스 가운데 오류(FP+FN)가 많은 클래스는" in text
    assert "가장 많은 세 클래스" not in text and "클래스별 오류(FP+FN)는" not in text


def test_judged_metric_values_keep_the_report_spelling_in_gap_sentences():
    """미달 값 0.69996 을 넷째 자리 '0.7' 로 찍으면 목표 0.7 과 같아 보인다 — 판정 대상이면 value_text."""
    f1 = MetricFact(metric_id="M4", display_name="F1 Score", value=0.69996, threshold=0.7, status="fail",
                    value_text="0.69996", is_core=True)
    p = MetricFact(metric_id="M2", display_name="Precision", value=0.79996, threshold=0.8, status="fail", value_text="0.79996")
    facts = {"selected_metrics": ["M2", "M3", "M4", "M22"],
             "average_gap": {"kind": "macro_vs_weighted", "macro_f1": 0.69996, "other_f1": 0.7123, "difference": 0.01234,
                             "difference_text": "0.0123", "reference_class": {"label": "a", "support": 9, "f1": 0.5}},
             "precision_recall_gap": {"precision": 0.79996, "recall": 0.6, "difference": 0.19996, "difference_text": "0.2",
                                      "flagged": True}}
    multi = {c.item_id: c.template_text for c in select_items(FactSheet(verdict="FAIL", score=0.0, metrics=[f1], facts=facts), None, "multiclass")}
    assert multi["8.2-07"].startswith("매크로 F1 은 0.69996,")
    binary = {c.item_id: c.template_text for c in select_items(FactSheet(verdict="FAIL", score=0.0, metrics=[p], facts=facts), None, "binary")}
    assert binary["8.2-05"].startswith("양성 클래스의 Precision 은 0.79996,")


@pytest.mark.parametrize("boundary, confident", [(1, 0), (0, 1)])
def test_error_nature_does_not_overstate_a_tiny_share(boundary, confident):
    fs = FactSheet(verdict="PASS", score=100.0, error_ratios=_er(boundary, confident, misclassified=200),
                   facts={"fixed_sentences": [{"id": "17", "section": "7.1", "text": "17"}]})
    text = next(c.template_text for c in select_items(fs, None, "binary") if c.item_id == "7.1-11")
    assert "비중이 크다" not in text and "달라지기 쉽다" not in text


def test_multiclass_error_nature_does_not_mention_a_decision_threshold():
    c71 = _sections(_request("multiclass", "FAIL"))["7.1"]["interpretation.confusion_analysis"]
    assert "경계 오류가 확신 오류보다 많다" in c71 and "임계값" not in c71


def test_ci_all_apart_does_not_say_stable_when_a_metric_is_near_target():
    m = MetricFact(metric_id="M4", display_name="F1 Score", value=0.512, threshold=0.5, status="pass", near_target=True,
                   ci_low=0.5051, ci_high=0.519, ci_contains_target=False, value_text="0.512", margin=0.012, margin_text="0.012")
    text = _ci_text(FactSheet(verdict="PASS", score=100.0, metrics=[m], facts={}))
    assert text == "판정 대상 지표의 95% 신뢰구간은 모두 목표값과 떨어져 있다."


def test_large_latency_values_are_plain_decimals_that_pass_the_number_check():
    req = _request("binary", "PASS")
    raw = json.loads(req.model_dump_json())
    raw["fact_sheet"]["latency"] = {"available": True, "mean": 1234.56789, "p50": 123456.75, "p95": 1234567.5,
                                    "p99": 0.00001, "unit": "ms"}
    req = NarrativeRequest.model_validate(raw)
    fs, task, ci, derived, refs = _materials(req)
    risks = _sections(req)["8.3"]["conclusion.risks"]
    assert "1234567.5ms" in risks and "e+" not in risks and "0.00001ms" in risks
    assert verify_grounding([risks], build_number_whitelist(fs, refs, derived, ci)).passed


def test_labels_only_in_facts_lists_are_names_not_code_words():
    raw = _raw("multiclass")
    raw["fact_sheet"]["facts"]["prediction_bias"]["rows"][0]["label"] = "tiger_shark"
    req = NarrativeRequest.model_validate(raw)
    fs, task, ci, derived, refs = _materials(req)
    parts = _sections(req)["7.1"]
    assert "tiger_shark" in parts["interpretation.confusion_analysis"]
    result = checks.check_section("7.1", parts, whitelist=build_number_whitelist(fs, refs, derived, ci), verdict=fs.verdict,
                                  conclusions=select_items(fs, ci, task), problem_ids=[], exempt=checks.name_exemptions(fs, ci))
    assert result.code_words == []


QC = {"class_label_descriptions": {"0": "합격품", "1": "불합격품(기준 미달)"}}


def test_class_descriptions_with_verdict_words_are_masked_for_the_contradiction_check():
    raw = json.loads(_request("binary", "PASS").model_dump_json())
    raw["client_inputs"] = QC
    req = NarrativeRequest.model_validate(raw)
    fs = req.fact_sheet
    masks = checks.verdict_masks(fs, req.client_inputs)
    assert "1(불합격품(기준 미달))" in masks
    text = _sections(req)["7.1"]["interpretation.confusion_analysis"]
    assert "양성 클래스 1(불합격품(기준 미달)) 기준으로" in text
    assert find_verdict_contradictions([text], "PASS")  # 가리지 않으면 판정 모순으로 읽힌다
    assert find_verdict_contradictions(checks.mask_names([text], masks), "PASS") == []
    # 이빨 — 가려도 진짜 미달 주장은 그대로 잡힌다.
    assert find_verdict_contradictions(checks.mask_names(["Accuracy 는 목표에 미달했다."], masks), "PASS")


async def test_generator_masks_class_descriptions_before_the_contradiction_check(make_section_client):
    raw = json.loads(_request("binary", "PASS").model_dump_json())
    raw["client_inputs"] = QC
    req = NarrativeRequest.model_validate(raw)
    named = "양성 클래스 1(불합격품(기준 미달))은 43건을 놓쳤다."  # FN 43 — D24(7.1)도 지킨다
    fake = make_section_client({"section_7_1": named})
    result = await generator.generate_sections(fake, req)
    assert result.section_sources["7.1"] == "llm"
    assert result.response.interpretation.confusion_analysis == named


def test_every_legacy_fallback_goes_through_the_wired_helper():
    """옛 경로의 규칙 문장 11곳이 모두 작업 유형·고객 입력을 넘기는 _rule_text 를 거친다(키 없음 경로만 보던 배선 테스트 보강)."""
    import inspect

    from app.narrative import legacy

    body = inspect.getsource(legacy._narrate_body)
    assert body.count("_rule_text(req, fs, benchmark_refs, derived, reason=") >= 11
    assert "build_fallback_narrative(" not in body
    assert inspect.getsource(legacy).count("build_fallback_narrative(") == 1  # _rule_text 안의 한 곳


@pytest.mark.parametrize("task", TASKS)
def test_fact_less_requests_follow_writing_rules_for_every_task_type(task):
    """facts·불균형 판단이 없는 옛 요청 — 멀티레이블 7.2 에 AI 용 설명('성적서가 따로 찍는다')이 새지 않는다."""
    raw = _raw(task)
    raw["fact_sheet"]["facts"] = None
    raw["fact_sheet"]["imbalance_effect"] = None
    req = NarrativeRequest.model_validate(raw)
    texts = _all_texts(_sections(req))
    for text in texts:
        for word in _AI_ONLY:
            assert word not in text, (word, text)
    if task == "multilabel":
        c72 = _sections(req)["7.2"]["interpretation.distribution_analysis"]
        assert c72.endswith("불균형 비율은 1.1031이다.")


def test_description_alone_is_masked_too():
    """AI 가 이름 없이 설명만 부를 수도 있다('불합격품(기준 미달)으로 분류된 표본') — 설명 글도 가린다."""
    raw = json.loads(_request("binary", "PASS").model_dump_json())
    raw["client_inputs"] = QC
    req = NarrativeRequest.model_validate(raw)
    masks = checks.verdict_masks(req.fact_sheet, req.client_inputs)
    text = "불합격품(기준 미달)으로 분류된 표본은 95건이다."
    assert find_verdict_contradictions([text], "PASS")
    assert find_verdict_contradictions(checks.mask_names([text], masks), "PASS") == []


def test_trimmed_small_class_list_ends_with_etc():
    raw = _raw("multiclass")
    facts = raw["fact_sheet"]["facts"]
    facts["low_support_classes"] = [{"label": "dog", "support": 12, "share": 0.06, "share_text": "0.06"}]
    facts["omitted"] = {"low_support_classes": 40}
    text = _sections(NarrativeRequest.model_validate(raw))["7.1"]["interpretation.confusion_analysis"]
    assert "그 밖에 표본이 작은 클래스는 dog 12건 등이다." in text
