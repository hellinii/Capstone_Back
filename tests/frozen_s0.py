"""tests/frozen_s0.py — 판 3 동작을 얼려 두는 정답(S0)의 '계산하는 쪽'. 테스트와 생성 스크립트가 같이 쓴다.

판 4 라운드에서 판 3 서술 경로가 실수로 바뀌면 기계가 잡게 한다. 정답 파일은 tests/golden_frozen/ 의 정적 파일이고,
여기 함수들이 지금 코드로 같은 값을 계산한다. 비교는 글자 그대로(dump(값) == 파일 내용)다 — 갱신 경로(UPDATE_*)는 없다.

- 입력도 얼린다: 계약 예시는 tests/golden_frozen/inputs/ 사본, 실제 발급본 요청·칸 글은 tests/data/edition3_issued/.
  tests/contract 는 판 4 라운드에서 바뀔 수 있어 읽지 않는다(바뀌면 이 정답이 말없이 따라가 버린다).
- 옛 골든(tests/golden)을 만든 테스트의 요청 조립 함수도 여기에 그대로 옮겨 두었다(원래 테스트가 바뀌어도 이쪽은 그대로).
- 저장소 밖 파일은 읽지 않는다(생성 스크립트의 발급본 추출 모드만 예외 — tests/generate_s0_goldens.py).

이 모듈은 pytest 수집 대상이 아니다(test_ 함수 없음).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.schemas import TaskType
from app.main import app
from app.narrative import checks, generator, legacy, prompt
from app.narrative.grounding import find_verdict_contradictions, verify_grounding
from app.narrative.jobs import parse_request
from app.narrative.schemas import ConfusionFact, DistributionFact, FactSheet, MetricFact, NarrativeRequest
from app.narrative.sections import SECTION_IDS, SECTIONS, SECTIONS_BY_ID

TESTS = Path(__file__).parent
FROZEN = TESTS / "golden_frozen"
INPUTS = FROZEN / "inputs"
ISSUED = TESTS / "data" / "edition3_issued"

# 판 3 발급본 6건 ↔ 서술 요청(모델명·작업 유형·판정·facts 가 요청과 같음을 추출 때 확인했다).
ISSUED_PAIRS = {
    "RPT-2026-0001": "bc_strong",
    "RPT-2026-0002": "bc_weak",
    "RPT-2026-0003": "digits_strong",
    "RPT-2026-0004": "digits_weak",
    "RPT-2026-0005": "yeast_strong",
    "RPT-2026-0006": "yeast_weak",
}

CONTRACT_CASES = ("binary", "multiclass", "multilabel")
# 합성 입력(tests/golden_frozen/inputs/fact_sheet_<이름>.json) — 계약·발급본이 지나지 않는 규칙 문장 갈래를 지나게 만든 것.
# 만든 방법은 tests/golden_frozen/README.md.
SYNTH_CASES = ("synth_inconclusive", "synth_conditional")
JOB_CASES = (*CONTRACT_CASES, *ISSUED_PAIRS.values(), *SYNTH_CASES)
LEGACY_PROMPT_CASES = ("legacy_main", "binary")

# 옛 골든(tests/golden/narrative_job_fallback_*)을 만든 테스트(test_s6_templates)의 고객 입력 — 그대로 옮김.
_DESCRIPTIONS = {
    "binary": {"0": "정상", "1": "불량"},
    "multiclass": {"cat": "고양이", "dog": "개", "bird": "새"},
    "multilabel": {"sports": "스포츠 기사", "tech": "기술 기사"},
}
_PURPOSE = "생산 라인 2 의 불량 검출"

# golden_utils 와 같은 비결정 경로 가림(평가 응답의 수행 시각·라이브러리 버전).
_REDACTED_PATHS = ("environment.evaluated_at", "environment.libraries")
_REDACTED = "<비결정적 값 — 존재만 고정>"


# ── 직렬화 — 정답 파일과 글자 그대로 비교하는 꼴 ─────────────────────────────────────


def dump(obj) -> str:
    """정답 파일의 글자. tests/golden(assert_golden)이 쓰던 꼴과 같다(끝 줄바꿈 없음)."""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def redact(o, path: str = ""):
    if any(path == p or path.startswith(p + ".") for p in _REDACTED_PATHS):
        return _REDACTED
    if isinstance(o, dict):
        return {k: redact(v, f"{path}.{k}" if path else str(k)) for k, v in o.items()}
    if isinstance(o, list):
        return [redact(v, f"{path}[{i}]") for i, v in enumerate(o)]
    return o


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def contract_raw(name: str) -> dict:
    return load_json(INPUTS / f"fact_sheet_{name}.json")


def issued_request_raw(name: str) -> dict:
    return load_json(ISSUED / "requests" / f"{name}.request.json")


def job_case_raw(case: str) -> dict:
    """작업 경로 정답의 입력 — 계약 3종·합성 입력은 inputs/ 사본, 그 밖은 발급본 요청 원문."""
    if case in CONTRACT_CASES or case in SYNTH_CASES:
        return contract_raw(case)
    assert case in ISSUED_PAIRS.values(), case
    return issued_request_raw(case)


# ── 1. 옛 골든의 재계산(tests/golden 의 narrative_*·validate_* 를 만든 것과 같은 입력) ──────────


def _sample_request() -> NarrativeRequest:
    """test_narrative_router._sample_request 사본."""
    fs = FactSheet(
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
        distribution=DistributionFact(class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.857),
    )
    return NarrativeRequest(task_type=TaskType.binary, report_purpose="internal", fact_sheet=fs)


def _problems_request() -> NarrativeRequest:
    """test_narrative_router._problems_request 사본."""
    fs = FactSheet(
        n_samples=200,
        dropped_rows=3,
        verdict="FAIL",
        score=66.7,
        metrics=[
            MetricFact(metric_id="M1", display_name="Accuracy", value=0.94, threshold=0.85, status="pass"),
            MetricFact(metric_id="M3", display_name="Recall", value=0.70, threshold=0.80, status="fail",
                       is_core=True, margin=-0.1),
            MetricFact(metric_id="M9", display_name="AUROC", value=0.88, threshold=0.80, status="pass"),
        ],
        confusion=ConfusionFact(labels=["0", "1"], matrix=[[120, 10], [15, 55]]),
        distribution=DistributionFact(class_distribution={"0": 130, "1": 70}, imbalance_ratio=1.857),
        f1_threshold_gain={"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37},
        imbalance_effect={
            "level": "affected", "imbalance_ratio": 1.857, "accuracy_minus_macro_f1": 0.12,
            "majority_minus_minority_f1": 0.18, "majority_class": "0", "minority_class": "1",
        },
        problems=[
            {"id": "P1", "kind": "core_fail", "priority": "HIGH", "title": "핵심 지표 미달 (Recall)",
             "source_section": "8.2", "recommend_section": "9.2",
             "evidence": {"Recall": {"value": 0.7, "target": 0.8}}, "note": None},
            {"id": "P2", "kind": "threshold_tuning", "priority": "MEDIUM", "title": "임계값 부적합",
             "source_section": "8.3", "recommend_section": "9.2",
             "evidence": {"max_f1": 0.8421, "current_f1": 0.7012, "gain": 0.1409, "best_threshold": 0.37},
             "note": None},
            {"id": "P3", "kind": "imbalance_effect", "priority": "MEDIUM", "title": "불균형 영향 있음 (다수 0 · 소수 1)",
             "source_section": "7.2", "recommend_section": "9.1",
             "evidence": {"level": "affected", "imbalance_ratio": 1.857, "accuracy_minus_macro_f1": 0.12,
                          "majority_minus_minority_f1": 0.18, "majority_class": "0", "minority_class": "1"},
             "note": "의뢰자가 편향 감소 노력을 '수행 안 함'으로 밝혔다."},
        ],
    )
    return NarrativeRequest(task_type=TaskType.binary, report_purpose="external", fact_sheet=fs)


def _legacy_post(client: TestClient, body: dict) -> dict:
    resp = client.post("/api/generate-narrative", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["meta"]["source"] == "fallback", data["meta"]
    return data


def _validate_post(client: TestClient, file_path: Path, data_json: str) -> dict:
    with open(file_path, "rb") as f:
        resp = client.post(
            "/api/validate-data",
            files={"file": (file_path.name, f.read(), "text/csv")},
            data={"data": data_json},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def legacy_route_goldens() -> dict[str, object]:
    """옛 경로(/api/generate-narrative) 규칙 문장 골든 3개 + 검증 골든 6개. 호출부가 anthropic_client 를 None 으로 둔다."""
    from router_cases import CASES, request_json  # tests/ 헬퍼(입력 CSV·매핑)

    client = TestClient(app)
    out: dict[str, object] = {
        "narrative_binary_fallback": _legacy_post(client, json.loads(_sample_request().model_dump_json())),
        "narrative_binary_fallback_problems": _legacy_post(client, json.loads(_problems_request().model_dump_json())),
        "narrative_legacy_main_fallback": _legacy_post(client, contract_raw("legacy_main")),
    }
    for task in CONTRACT_CASES:
        c = CASES[task]
        out[f"validate_{task}_clean"] = _validate_post(client, c["csv"], request_json(task))
        out[f"validate_{task}_errors"] = _validate_post(client, c["csv_errors"], request_json(task))
    return {k: redact(v) for k, v in out.items()}


def _job_fallback_request(task: str) -> NarrativeRequest:
    """test_s6_templates._request(task, "FAIL", latency=False) 와 같다 — 계약 예시 + 클래스 설명·모델 용도."""
    raw = contract_raw(task)
    raw["client_inputs"] = {"class_label_descriptions": _DESCRIPTIONS[task], "model": {"모델 용도": _PURPOSE}}
    return NarrativeRequest.model_validate(raw)


async def _templates(req: NarrativeRequest) -> dict:
    result = await generator.generate_sections(None, req)
    assert result.response.meta.reason == "no_key"
    return {
        "response": result.response.model_dump(mode="json"),
        "section_sources": result.section_sources,
        "section_models": result.section_models,
    }


async def job_fallback_goldens() -> dict[str, object]:
    """tests/golden/narrative_job_fallback_* 와 같은 계산."""
    return {f"narrative_job_fallback_{t}": await _templates(_job_fallback_request(t)) for t in CONTRACT_CASES}


# ── 3. 판 3 작업 경로·옛 경로의 새 정답 ─────────────────────────────────────────────

_FIELD_TO_SECTION = {f: s.id for s in SECTIONS for f in s.ai_fields}


class _RecordingClient:
    """작업 경로가 실제로 보내는 요청을 그대로 받아 적는 가짜 클라이언트. 응답은 거절(refusal) — 다시 부르지 않고
    그 장을 규칙 문장으로 끝내므로, 기록되는 것은 장별 첫 호출 그대로다."""

    def __init__(self):
        self.sent: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            model="recording", stop_reason="refusal", stop_details=None, content=[],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, iterations=None),
        )


async def job_prompts(req: NarrativeRequest) -> dict:
    """장별로 실제 보낸 system·user·응답 스키마(가짜 클라이언트가 받아 적음) + 칸별 다시 만들기 user 메시지."""
    rec = _RecordingClient()
    await generator.generate_sections(rec, req)
    chapters: dict[str, dict] = {}
    for kw in rec.sent:
        schema = kw["output_config"]["format"]["schema"]
        sids = sorted({_FIELD_TO_SECTION[f] for f in schema["properties"]}, key=SECTION_IDS.index)
        chapter = SECTIONS_BY_ID[sids[0]].chapter
        assert chapter not in chapters, chapter
        assert [m["role"] for m in kw["messages"]] == ["user"]
        chapters[chapter] = {
            "sections": sids,
            "system": kw["system"],
            "user": kw["messages"][0]["content"],
            "schema": schema,
        }
    _, _, _, _, ctx, _, _ = generator._prepare(req)
    regenerate = {sid: prompt.build_user_message(ctx, (sid,), regenerate=True) for sid in SECTION_IDS}
    return {"chapters": chapters, "regenerate_user": regenerate}


def fact_view(req: NarrativeRequest) -> dict:
    """작업 경로의 사실 시트 보기(지시문에 실리는 글)·고객 입력 보기·숫자 허용 목록(정렬)."""
    _, _, whitelist, _, ctx, _, _ = generator._prepare(req)
    return {
        "fact_view_text": json.dumps(ctx.view, ensure_ascii=False, separators=(", ", ":")),
        "client_view_text": (
            json.dumps(ctx.client_view, ensure_ascii=False, separators=(", ", ":")) if ctx.client_view else None
        ),
        "trimmed": list(ctx.trimmed),
        "dropped_client_keys": list(ctx.dropped_client_keys),
        "whitelist": sorted(whitelist),
    }


def legacy_prompts(raw: dict) -> dict:
    """옛 경로(/api/generate-narrative)가 AI 에 보낼 system·user·응답 스키마와 그 경로의 숫자 허용 목록."""
    req = NarrativeRequest.model_validate(raw)
    fs = req.fact_sheet
    derived, refs, whitelist = legacy._prepare_facts(fs, req.task_type.value, req.client_inputs)
    system, user, dropped = legacy.build_prompts_within_limit(req, fs, derived, refs)
    return {
        "system": system,
        "user": user,
        "dropped": dropped,
        "schema": legacy.build_response_schema(fs.problems),
        "whitelist": sorted(whitelist),
    }


async def edition3_goldens() -> dict[str, object]:
    """golden_frozen/edition3/ 의 정답 — 이름(확장자 없는 상대 경로) → 값."""
    out: dict[str, object] = {}
    for case in JOB_CASES:
        req, trimmed = parse_request(copy.deepcopy(job_case_raw(case)))  # 작업 경로의 요청 해석 그대로
        templates = await _templates(req)
        templates["parse_trimmed"] = trimmed
        out[f"edition3/{case}.templates"] = templates
        out[f"edition3/{case}.job_prompts"] = await job_prompts(req)
        out[f"edition3/{case}.fact_view"] = fact_view(req)
    for case in LEGACY_PROMPT_CASES:
        out[f"edition3/{case}.legacy_prompts"] = legacy_prompts(contract_raw(case))
    return out


# ── 4. 실제 판 3 발급본 AI 글의 칸별 검문 판정 ──────────────────────────────────────


def issued_slots(report_no: str) -> dict:
    return load_json(ISSUED / "slots" / f"{report_no}.json")


def _check_one(sid: str, parts: dict, req: NarrativeRequest, prepared) -> dict:
    """generator.generate_sections 의 accept() 와 같은 인자로 check_section 을 부르고, 숫자 검문·최종 판정 모순도 적는다."""
    fs = req.fact_sheet
    _, _, whitelist, conclusions, _, exempt, masks = prepared
    problem_ids = [p.id for p in fs.problems]
    aligned = generator._align(parts, fs.problems, fill_missing=False, edition=3)[0] if sid == "9" else parts
    rec_parts = generator._rec_dicts(aligned)
    result = checks.check_section(
        sid, rec_parts, whitelist=whitelist, verdict=fs.verdict, conclusions=conclusions,
        problem_ids=problem_ids, exempt=exempt, masks=masks, labels=checks.numeric_labels(fs), edition=3,
    )
    texts = [t for t in checks.section_texts(sid, rec_parts) if isinstance(t, str)]
    grounding = verify_grounding(texts, whitelist)
    return {
        "check": result.as_record(),
        "hard_failures": result.hard_failures,
        "passed": result.passed,
        "grounding": {"passed": grounding.passed, "violations": grounding.violations, "checked": grounding.checked},
        "final_verdict_contradictions": find_verdict_contradictions(checks.mask_names(texts, masks), fs.verdict),
    }


def issued_checks(report_no: str) -> dict:
    slots = issued_slots(report_no)
    assert slots["report_no"] == report_no and slots["request"] == ISSUED_PAIRS[report_no]
    req, trimmed = parse_request(copy.deepcopy(issued_request_raw(slots["request"])))
    assert req.task_type.value == slots["task_type"]
    prepared = generator._prepare(req)
    return {
        "report_no": report_no,
        "request": slots["request"],
        "parse_trimmed": trimmed,
        "sections": {sid: _check_one(sid, slots["parts"][sid], req, prepared) for sid in SECTION_IDS},
    }


def issued_goldens() -> dict[str, object]:
    return {f"edition3/issued_checks_{no}": issued_checks(no) for no in ISSUED_PAIRS}


# ── 발급본에서 칸 글 뽑기(생성 스크립트의 추출 모드만 — 테스트는 tests/data 사본만 읽는다) ─────────


def slots_from_content(content: dict) -> dict[str, dict]:
    """판 3 저장본(FinalReportData, camelCase)의 서술 칸 → 칸 ID → {응답 필드(점 표기): 값}(작업 경로 _extract 모양)."""
    interp, concl, recn = content["interpretation"], content["conclusion"], content["recommendationNarrative"]
    recs = [
        {"problem_id": r["problemId"], "category": r["category"], "action": r["action"],
         "expected_impact": r["expectedImpact"]}
        for r in content["recommendations"]
    ]
    return {
        "7.1": {"interpretation.confusion_analysis": interp["confusionAnalysis"]},
        "7.2": {"interpretation.distribution_analysis": interp["distributionAnalysis"]},
        "7.3": {"interpretation.additional_observations": interp["additionalObservations"]},
        "8.1": {"conclusion.benchmark": concl["benchmark"]},
        "8.2": {"conclusion.narrative": concl["narrative"]},
        "8.3": {"conclusion.risks": concl["risks"]},
        "9": {
            "recommendation_narrative.data_quality": recn["dataQuality"],
            "recommendation_narrative.model_ops": recn["modelOps"],
            "recommendations": recs,
        },
    }


# ── 비교 ─────────────────────────────────────────────────────────────────────────


def assert_frozen(name: str, value) -> None:
    """tests/golden_frozen/<name>.json 과 글자 그대로 같아야 한다. 파일이 없으면 실패(만들지 않는다)."""
    import difflib

    path = FROZEN / f"{name}.json"
    assert path.is_file(), f"고정 정답이 없다: {path} — 테스트는 정답을 만들지 않는다(README 참고)."
    expected = path.read_text(encoding="utf-8")
    actual = dump(value)
    if actual != expected:
        diff = list(difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), "정답(golden_frozen)", "지금 코드", lineterm="", n=1,
        ))
        # 지시문 user 처럼 한 줄이 아주 긴 값이 있어 줄마다 앞부분만 보인다(어디가 다른지는 줄 번호로 찾는다).
        shown = "\n".join(line if len(line) <= 300 else line[:300] + " …" for line in diff[:80])
        shown += "\n…" if len(diff) > 80 else ""
        raise AssertionError(
            f"판 3 고정 정답과 다르다: {name}. 정답을 다시 만들지 말고 코드를 고친다(golden_frozen/README.md).\n{shown}"
        )
