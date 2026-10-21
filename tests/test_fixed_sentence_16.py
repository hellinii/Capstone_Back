"""test_fixed_sentence_16.py — 고정 문구 16(8.1 표 아래 '내부 참조 기준치가 없어 …', 사용자 결정 2026-10-22).

- 일부 지표만 기준이 없으면: 지시문의 '성적서가 따로 찍는 문장'에 16 원문(되풀이 금지). 8.1 은 평소대로 AI.
- 서술 지표 전부가 기준이 없고 성적서가 16 을 찍으면: 8장 호출에서 8.1 을 빼고 8.1 = 빈 글·출처 fixed
  (문제 0개의 9칸과 같은 꼴). 규칙 템플릿(작업 경로)도 빈 글.
- facts 없는 옛 요청·옛 경로(build_fallback_narrative)는 지금 문장("비교 가능한 … 생략한다") 그대로.
조건마다 양쪽 갈래(있음/없음)를 함께 시험한다. 사실 시트는 계약 예시(tests/contract)를 고쳐 쓴다 — 예시가 16 을 싣든
안 싣든 여기서 고정 문장 목록을 직접 정한다.
"""
import json
from pathlib import Path

from app.narrative import checks, fallback, generator, prompt
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.required_items import benchmark_printed_only, code_prints, select_items
from app.narrative.schemas import NarrativeRequest
from app.narrative.sections import CHAPTERS
from tests.conftest import requested_fields

CONTRACT = Path(__file__).parent / "contract"

ALL_LACK = "선택한 지표에 내부 참조 기준치가 없어 벤치마크 비교를 하지 않았다."
PARTIAL = "다음 지표는 내부 참조 기준치가 없어 벤치마크 비교를 하지 않았다: F-beta Score."
OLD_SENTENCE = "비교 가능한 내부 참조 기준치가 없어 벤치마크 비교를 생략한다."
D24_OK = "의뢰자가 위험하다고 밝힌 FN 은 43건이다."


def _raw() -> dict:
    return json.loads((CONTRACT / "fact_sheet_binary.json").read_text())


def _without_16(raw: dict) -> dict:
    facts = raw["fact_sheet"]["facts"]
    facts["fixed_sentences"] = [s for s in facts["fixed_sentences"] if s["id"] != "16"]
    return raw


def _all_lack(raw: dict | None = None, *, sentence: bool = True) -> dict:
    """서술 지표를 기준표에 없는 F-beta Score 하나로 줄인다(이진 기준표: Accuracy·…·MCC)."""
    raw = _without_16(raw or _raw())
    fs = raw["fact_sheet"]
    fs["metrics"] = [m for m in fs["metrics"] if m["display_name"] == "F-beta Score"]
    assert fs["metrics"], "예시에 F-beta Score 가 있어야 한다"
    fs["facts"]["selected_metrics"] = [m["metric_id"] for m in fs["metrics"]]
    if sentence:
        fs["facts"]["fixed_sentences"].append({"id": "16", "section": "8.1", "text": ALL_LACK})
    return raw


def _req(raw: dict) -> NarrativeRequest:
    return NarrativeRequest.model_validate(raw)


def _parts(req: NarrativeRequest):
    fs, task = req.fact_sheet, req.task_type.value
    refs = build_benchmark_refs(task, fs.metrics)
    derived = compute_derived(fs, task_type=task)
    return fs, refs, derived, select_items(fs, req.client_inputs, task, derived=derived, benchmark_refs=refs)


def _printed(req: NarrativeRequest) -> bool:
    fs, refs, _, conclusions = _parts(req)
    return benchmark_printed_only(fs, refs, conclusions)


def _client(make_section_client):
    return make_section_client({"section_7_1": D24_OK})


def _fields(fake) -> list[list[str]]:
    return [requested_fields(c.kwargs) for c in fake.beta.messages.create.call_args_list]


# ── 판정 함수(required_items.benchmark_printed_only) ────────────────────────────

def test_all_lack_with_sentence_16_is_printed_only():
    assert _printed(_req(_all_lack()))


def test_all_lack_without_sentence_16_is_not_printed_only():
    """facts 는 있는데 16 을 싣지 않았다(프론트가 찍지 않음) — 백엔드가 지금 문장을 쓴다."""
    assert not _printed(_req(_all_lack(sentence=False)))


def test_partial_lack_with_sentence_16_is_not_printed_only():
    """기준표에 있는 지표가 하나라도 있으면(참조 표가 비지 않음) 8.1 은 평소대로."""
    raw = _without_16(_raw())
    raw["fact_sheet"]["facts"]["fixed_sentences"].append({"id": "16", "section": "8.1", "text": PARTIAL})
    req = _req(raw)
    fs, refs, _, conclusions = _parts(req)
    assert refs and any(c.item_id == "8.1-03" for c in conclusions)
    assert not benchmark_printed_only(fs, refs, conclusions)


def test_no_facts_is_not_printed_only():
    raw = _all_lack()
    raw["fact_sheet"]["facts"] = None
    assert not _printed(_req(raw))


def test_trimmed_sentence_list_counts_as_printed():
    """프론트가 facts 상한에 맞춰 고정 문장 목록을 비웠어도(omitted.fixed_sentences) 화면은 16 을 찍는다."""
    raw = _all_lack()
    facts = raw["fact_sheet"]["facts"]
    facts["fixed_sentences"] = []
    facts["omitted"] = {"fixed_sentences": 3}
    req = _req(raw)
    assert code_prints(req.fact_sheet, "16")
    assert _printed(req)


def test_trimmed_list_with_no_narrative_metric_is_not_printed_only():
    """문장 목록을 비우면 code_prints 는 어떤 번호든 참이다 — 서술 지표가 없으면(8.1-03 없음) 프론트는 16 을 찍지 않았다."""
    raw = _all_lack()
    facts = raw["fact_sheet"]["facts"]
    facts["fixed_sentences"] = []
    facts["omitted"] = {"fixed_sentences": 3}
    req = _req(raw)
    fs, refs, _, conclusions = _parts(req)
    assert code_prints(fs, "16") and not refs
    without = [c for c in conclusions if c.item_id != "8.1-03"]
    assert len(without) == len(conclusions) - 1
    assert not benchmark_printed_only(fs, refs, without)


def test_trimmed_list_without_facts_omission_is_not_printed():
    raw = _all_lack(sentence=False)
    raw["fact_sheet"]["facts"]["omitted"] = {}
    assert not code_prints(_req(raw).fact_sheet, "16")


# ── 규칙 템플릿 ────────────────────────────────────────────────────────────────

def test_job_template_8_1_is_empty_when_printed_only():
    req = _req(_all_lack())
    fs, refs, derived, conclusions = _parts(req)
    out = fallback.section_template("8.1", fs, refs, derived, task_type="binary", conclusions=conclusions)
    assert out == {"conclusion.benchmark": ""}


def test_job_template_8_1_keeps_old_sentence_when_not_printed():
    req = _req(_all_lack(sentence=False))
    fs, refs, derived, conclusions = _parts(req)
    out = fallback.section_template("8.1", fs, refs, derived, task_type="binary", conclusions=conclusions)
    assert out == {"conclusion.benchmark": OLD_SENTENCE}


def test_legacy_fallback_keeps_old_sentence_even_with_sentence_16():
    """옛 경로(build_fallback_narrative)는 16 이 실려도 지금 문장 그대로."""
    req = _req(_all_lack())
    fs, refs, derived, _ = _parts(req)
    resp = fallback.build_fallback_narrative(fs, refs, derived, "no_key", task_type="binary")
    assert resp.conclusion.benchmark == OLD_SENTENCE


def test_empty_8_1_is_still_rejected_by_the_check():
    """빈 8.1 은 AI 칸 검사에서는 여전히 빈 칸 위반이다 — 허용은 8.1 을 부르지 않는 경우뿐(그때는 검사하지 않는다)."""
    req = _req(_all_lack())
    _, _, whitelist, conclusions, *_ = generator._prepare(req)
    result = checks.check_section(
        "8.1", {"conclusion.benchmark": ""}, whitelist=whitelist, verdict=req.fact_sheet.verdict,
        conclusions=conclusions, problem_ids=[], exempt=set(),
    )
    assert result.format == ["empty"]


# ── 작업 경로 생성기 ───────────────────────────────────────────────────────────

async def test_generator_skips_8_1_when_printed_only(make_section_client):
    fake = _client(make_section_client)
    gen = await generator.generate_sections(fake, _req(_all_lack()))
    fields = _fields(fake)
    assert ["section_8_2", "section_8_3"] in fields  # 8장 호출은 남고 8.1 만 빠진다
    assert all("section_8_1" not in f for f in fields)
    assert gen.section_sources["8.1"] == "fixed"
    assert gen.response.conclusion.benchmark == ""
    assert gen.trace["section_8_1_fixed"] is True
    assert gen.trace["sections"]["8.1"]["http_calls"] == 0
    # 내부 기록의 결과·사유도 고정 문구 16 몫으로 남는다(세션 B 변이 M27·M29).
    assert gen.trace["sections"]["8.1"]["outcome"] == "fixed"
    assert gen.trace["sections"]["8.1"]["reason"] == "no_benchmark"
    assert gen.section_models["8.1"] is None and gen.section_models["8.2"] is not None  # 고정 문구 칸은 모델 없음
    # 다른 칸은 그대로 AI — 8.1 의 fixed 는 전체 출처를 규칙 문장으로 떨어뜨리지 않는다.
    assert gen.section_sources["8.2"] == gen.section_sources["8.3"] == "llm"
    assert gen.response.meta.source == "llm"


async def test_generator_calls_8_1_when_not_printed(make_section_client):
    fake = _client(make_section_client)
    gen = await generator.generate_sections(fake, _req(_all_lack(sentence=False)))
    assert ["section_8_1", "section_8_2", "section_8_3"] in _fields(fake)
    assert gen.section_sources["8.1"] == "llm"
    assert "section_8_1_fixed" not in gen.trace


async def test_generator_without_key_keeps_8_1_fixed_and_empty():
    gen = await generator.generate_sections(None, _req(_all_lack()))
    assert gen.section_sources["8.1"] == "fixed"
    assert gen.response.conclusion.benchmark == ""
    assert gen.section_sources["8.2"] == "fallback"
    # 키 없이 끝까지 가도(finish 의 규칙 문장 채우기) 8.1 의 결과·사유는 덮이지 않는다.
    assert gen.trace["sections"]["8.1"]["outcome"] == "fixed"
    assert gen.trace["sections"]["8.1"]["reason"] == "no_benchmark"


async def test_generator_without_key_and_without_facts_keeps_old_sentence():
    raw = _all_lack()
    raw["fact_sheet"]["facts"] = None
    gen = await generator.generate_sections(None, _req(raw))
    assert gen.section_sources["8.1"] == "fallback"
    assert gen.response.conclusion.benchmark == OLD_SENTENCE


# ── 지시문: 일부만 기준이 없을 때 16 원문을 '성적서가 따로 찍는 문장'에 ──────────

def _chapter_8_message(raw: dict) -> str:
    req = _req(raw)
    ctx = generator._prepare(req)[4]
    return prompt.build_user_message(ctx, CHAPTERS["8"])


def test_prompt_quotes_sentence_16_when_sent():
    raw = _without_16(_raw())
    raw["fact_sheet"]["facts"]["fixed_sentences"].append({"id": "16", "section": "8.1", "text": PARTIAL})
    msg = _chapter_8_message(raw)
    head = "성적서가 따로 찍는 문장(되풀이하지 않는다):"
    assert head in msg
    assert f'"{PARTIAL}"' in msg[msg.index(head):]
    assert prompt._PRINTED_ELSEWHERE["8.1-03"] not in msg


def test_prompt_falls_back_to_description_without_sentence_16():
    msg = _chapter_8_message(_without_16(_raw()))
    assert prompt._PRINTED_ELSEWHERE["8.1-03"] in msg
    assert PARTIAL not in msg
