"""test_edition3_frozen.py — 판 4 라운드 S0: 판 3 서술 동작을 정적 정답으로 고정한다(계획 2.5-3).

정답은 S0 에서 운영 main 코드로 떴다(tests/golden_frozen/README.md). 비교는 글자 그대로이고 갱신 경로가 없다 —
파일이 없으면 실패한다. 판 4 코드를 넣어도 판 3 요청(판 표시가 없는 요청)의 결과는 한 글자도 바뀌면 안 된다.

계약 예시 3종(이진·다중·멀티 — 사본 tests/golden_frozen/inputs), 실제 발급본 요청 6건(bc_strong·bc_weak·digits_strong·
digits_weak·yeast_strong·yeast_weak — tests/data/edition3_issued), 합성 입력 2개(synth_inconclusive·synth_conditional —
inputs, 계약·발급본이 지나지 않는 규칙 문장 갈래용)에 대해
a. 규칙 문장 7칸(키 없음 → generate_sections(None, req), 작업 경로의 요청 해석 parse_request 그대로)
b. 장별로 실제 보낸 system·user·응답 스키마(가짜 클라이언트가 받아 적음) + 칸별 다시 만들기 user 메시지
c. 사실 시트 보기 글·고객 입력 보기 글·숫자 허용 목록(정렬)
d. 옛 경로(/api/generate-narrative) 지시문 — 옛 main 모양 예시와 이진 예시
그리고 실제 판 3 발급본 6건(RPT-2026-0001~0006)의 AI 칸 글에 대한 칸별 검문 판정(숫자 검문·check_section).
"""
import pytest

import frozen_s0 as fz
from app.narrative.sections import SECTION_IDS


_CACHE: dict = {}


async def _computed() -> dict:
    """한 번 계산해 이 모듈의 테스트가 같이 쓴다(계산은 결정론적이다)."""
    if not _CACHE:
        _CACHE.update(await fz.edition3_goldens())
    return _CACHE


EXPECTED_NAMES = sorted(
    [f"edition3/{c}.{kind}" for c in fz.JOB_CASES for kind in ("templates", "job_prompts", "fact_view")]
    + [f"edition3/{c}.legacy_prompts" for c in fz.LEGACY_PROMPT_CASES]
)


async def test_edition3_golden_names():
    computed = await _computed()
    assert sorted(computed) == EXPECTED_NAMES


@pytest.mark.parametrize("name", EXPECTED_NAMES)
async def test_edition3_output_equals_frozen(name):
    computed = await _computed()
    fz.assert_frozen(name, computed[name])


def test_frozen_edition3_files_are_all_compared():
    on_disk = sorted(f"edition3/{p.stem}" for p in (fz.FROZEN / "edition3").glob("*.json"))
    issued = [f"edition3/issued_checks_{no}" for no in fz.ISSUED_PAIRS]
    assert on_disk == sorted(EXPECTED_NAMES + issued)


# 이빨 — 정답이 무엇을 담았는지 몇 가지를 손으로 단정한다(정답이 비어 있거나 엉뚱한 것을 얼리지 않았다).


def test_frozen_templates_cover_all_seven_slots():
    for case in fz.JOB_CASES:
        data = fz.load_json(fz.FROZEN / "edition3" / f"{case}.templates.json")
        assert list(data["section_sources"]) == list(SECTION_IDS)
        assert data["response"]["meta"]["reason"] == "no_key"
        assert data["response"]["conclusion"]["narrative"].strip(), case


def test_frozen_job_prompts_hold_the_system_prompt_and_chapters():
    from app.narrative.prompt import SYSTEM_PROMPT

    for case in fz.JOB_CASES:
        data = fz.load_json(fz.FROZEN / "edition3" / f"{case}.job_prompts.json")
        assert set(data["chapters"]) <= {"7", "8", "9"} and {"7", "8"} <= set(data["chapters"]), case
        for ch in data["chapters"].values():
            assert ch["system"] == SYSTEM_PROMPT
            assert ch["user"].startswith("[이번 호출]\n이번 호출은 ")
            assert "[사실 시트]\n" in ch["user"]
        assert list(data["regenerate_user"]) == list(SECTION_IDS)
    bc = fz.load_json(fz.FROZEN / "edition3" / "bc_strong.job_prompts.json")
    assert "[고객 입력 — 의뢰자가 제공한 정보]" in bc["chapters"]["7"]["user"]


def test_frozen_fact_view_whitelist_is_sorted_and_used():
    for case in fz.JOB_CASES:
        data = fz.load_json(fz.FROZEN / "edition3" / f"{case}.fact_view.json")
        wl = data["whitelist"]
        assert wl == sorted(wl) and len(wl) > 50, case
        assert data["fact_view_text"] in fz.load_json(fz.FROZEN / "edition3" / f"{case}.job_prompts.json")["chapters"]["7"]["user"]


# ── 실제 판 3 발급본 AI 글의 칸별 검문 판정 ─────────────────────────────────────────


@pytest.mark.parametrize("report_no", list(fz.ISSUED_PAIRS))
def test_issued_edition3_slot_checks_equal_frozen(report_no):
    fz.assert_frozen(f"edition3/issued_checks_{report_no}", fz.issued_checks(report_no))


def test_issued_slot_texts_are_real_ai_text():
    """칸 글은 발급본에서 옮긴 AI 글이다(출처 llm) — 규칙 문장을 검문한 것이 아니다."""
    for no in fz.ISSUED_PAIRS:
        slots = fz.issued_slots(no)
        assert slots["narrative_sections"] == dict.fromkeys(SECTION_IDS, "llm"), no
        assert list(slots["parts"]) == list(SECTION_IDS)
        assert all(slots["parts"][sid] for sid in SECTION_IDS)


def test_issued_checks_detect_a_changed_number():
    """검문 판정이 헛돌지 않는다 — 발급본 7.1 글의 숫자 하나를 허용 목록 밖 숫자로 바꾸면 숫자 검문에 걸린다."""
    import copy

    from app.narrative import generator
    from app.narrative.jobs import parse_request

    slots = fz.issued_slots("RPT-2026-0001")
    req, _ = parse_request(copy.deepcopy(fz.issued_request_raw("bc_strong")))
    prepared = generator._prepare(req)
    parts = dict(slots["parts"]["7.1"])
    ok = fz._check_one("7.1", parts, req, prepared)
    assert ok["grounding"]["passed"] and ok["grounding"]["checked"] > 5
    text = parts["interpretation.confusion_analysis"]
    assert "171건" in text
    parts["interpretation.confusion_analysis"] = text.replace("171건", "173건")
    bad = fz._check_one("7.1", parts, req, prepared)
    assert bad["grounding"]["violations"] == ["173"] and "numbers" in bad["hard_failures"]


# ── 정답이 덮는 갈래 — 계약·발급본·합성 입력이 각자 맡은 규칙 문장이 정답 안에 실제로 있다 ─────────────


def test_job_cases_cover_contract_issued_and_synthetic_inputs():
    assert fz.JOB_CASES == (*fz.CONTRACT_CASES, *fz.ISSUED_PAIRS.values(), *fz.SYNTH_CASES)
    for case in fz.SYNTH_CASES:
        assert (fz.INPUTS / f"fact_sheet_{case}.json").is_file(), case


def _templates_text(case: str) -> str:
    return (fz.FROZEN / "edition3" / f"{case}.templates.json").read_text(encoding="utf-8")


def _prompts_text(case: str) -> str:
    return (fz.FROZEN / "edition3" / f"{case}.job_prompts.json").read_text(encoding="utf-8")


def test_synthetic_inputs_reach_their_branches():
    """합성 입력이 노린 갈래를 정답이 실제로 담았다(입력을 고치다 갈래를 놓치면 여기서 걸린다)."""
    cond = fz.load_json(fz.FROZEN / "edition3" / "synth_conditional.templates.json")["response"]
    # 근소 차이 지표가 있고 모든 구간이 목표와 떨어짐(required_items._ci 의 근소 갈래)
    assert "판정 대상 지표의 95% 신뢰구간은 모두 목표값과 떨어져 있다." in cond["conclusion"]["narrative"]
    assert "가능성은 낮다" not in cond["conclusion"]["narrative"]
    assert "'조건부 합격'" in cond["conclusion"]["narrative"]
    assert [r["category"] for r in cond["recommendations"]] == ["보조 지표 개선", "판정 안정성 확인", "정답 라벨 검증"]
    inc = fz.load_json(fz.FROZEN / "edition3" / "synth_inconclusive.templates.json")["response"]
    assert "(원인: 핵심 지표 산출 실패 — AUROC)" in inc["conclusion"]["narrative"]
    assert "다만 종합 판정이 '판정 불가'여서 판정에 따른 위험은 확인하지 못했다." in inc["conclusion"]["risks"]
    assert inc["recommendations"][0]["category"] == "판정 조건 보완"
    assert "판정 불가 원인은 핵심 지표 산출 실패(AUROC)이다." in _prompts_text("synth_inconclusive")
    assert "목표에 미달한 비핵심 지표는 Accuracy이다." in _prompts_text("synth_inconclusive")


def test_issued_cases_hold_the_low_risk_ci_sentence():
    """'가능성은 낮다' 갈래(모든 구간이 떨어지고 근소 지표 없음)는 발급본 요청이 덮는다."""
    hits = [c for c in fz.ISSUED_PAIRS.values() if "판정이 달라질 가능성은 낮다." in _templates_text(c)]
    assert hits, "발급본 정답에 '가능성은 낮다' 문장이 없다"
