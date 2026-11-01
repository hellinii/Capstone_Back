"""tests/test_narrative_digest.py — 서술 글 지문의 정규화(app.narrative.digest), 심판 라운드 J9(계획 14-9b).

작업 쪽(from_result)은 프론트 mapNarrativeResult 가 하는 옮김을 그대로 흉내 내고, content 쪽(from_content)은 저장된 값을 그대로 읽는다.
규칙 한 줄마다 "프론트가 저장할 content" 를 손으로 적어 같은 지문인지, 그리고 그 규칙이 없으면 달라지는지를 본다.
"""
from __future__ import annotations

import copy

import pytest

from app.narrative import digest

RESULT = {
    "interpretation": {"confusion_analysis": "혼동 글", "distribution_analysis": "분포 글", "additional_observations": "추가 글"},
    "conclusion": {"benchmark": "8.1 글", "narrative": "총평 글", "risks": "위험 글"},
    "recommendation_narrative": {"data_quality": "데이터 글", "model_ops": "운영 글"},
    "recommendations": [
        {"problem_id": "P1", "priority": "HIGH", "category": "데이터", "action": "고친다", "expected_impact": "좋아진다"},
        {"problem_id": "P2", "priority": "LOW", "category": "모델", "action": "본다", "expected_impact": "안다"},
    ],
    "section_sources": {"7.1": "llm"}, "section_models": {"7.1": "m"}, "judge": {"gate": "passed"},
}

CONTENT = {
    "contentSchema": 4,
    "interpretation": {"confusionAnalysis": "혼동 글", "distributionAnalysis": "분포 글", "additionalObservations": "추가 글"},
    "conclusion": {"verdict": "FAIL", "score": 1, "benchmark": "8.1 글", "narrative": "총평 글", "risks": "위험 글"},
    "recommendationNarrative": {"dataQuality": "데이터 글", "modelOps": "운영 글"},
    "recommendations": [
        {"priority": "HIGH", "category": "데이터", "action": "고친다", "expectedImpact": "좋아진다", "problemId": "P1"},
        {"priority": "LOW", "category": "모델", "action": "본다", "expectedImpact": "안다", "problemId": "P2"},
    ],
}


def _same(result, content) -> bool:
    return digest.of_result(result) == digest.of_content(content)


def test_plain_mapping_matches():
    assert _same(RESULT, CONTENT)
    assert len(digest.of_result(RESULT)) == 64


def test_non_narrative_keys_are_ignored_on_both_sides():
    r = {**RESULT, "section_sources": {}, "judge": None}
    c = {**CONTENT, "conclusion": {**CONTENT["conclusion"], "verdict": "PASS"}, "title": "x", "judge": {}}
    assert _same(r, c)


def test_missing_result_text_is_empty_like_front():
    """프론트 `?? ""` — 옛 서버처럼 7.3 칸이 없거나 null 이면 빈 글로 저장된다."""
    r = copy.deepcopy(RESULT)
    r["interpretation"]["additional_observations"] = None
    c = copy.deepcopy(CONTENT)
    c["interpretation"]["additionalObservations"] = ""
    assert _same(r, c)
    del r["interpretation"]["additional_observations"]
    assert _same(r, c)
    c["interpretation"]["additionalObservations"] = None  # content 쪽은 고치지 않는다 — null 은 빈 글과 다르다
    assert not _same(r, c)


@pytest.mark.parametrize("given, stored", [("high", "HIGH"), ("Low", "LOW"), ("URGENT", "MEDIUM"), (None, "MEDIUM"), ("", "MEDIUM")])
def test_priority_like_front(given, stored):
    r = copy.deepcopy(RESULT)
    r["recommendations"][0]["priority"] = given
    c = copy.deepcopy(CONTENT)
    c["recommendations"][0]["priority"] = stored
    assert _same(r, c)
    c["recommendations"][0]["priority"] = given
    if given != stored:
        assert not _same(r, c)


def test_problem_id_trimmed_and_blank_dropped_like_front():
    r = copy.deepcopy(RESULT)
    r["recommendations"][0]["problem_id"] = "  P1 "
    assert _same(r, CONTENT)
    r["recommendations"][0]["problem_id"] = "   "
    c = copy.deepcopy(CONTENT)
    del c["recommendations"][0]["problemId"]
    assert _same(r, c)
    c["recommendations"][0]["problemId"] = ""  # 프론트는 빈 번호면 키를 두지 않는다
    assert not _same(r, c)


def test_missing_row_text_is_empty_like_front():
    r = copy.deepcopy(RESULT)
    r["recommendations"][1]["category"] = None
    c = copy.deepcopy(CONTENT)
    c["recommendations"][1]["category"] = ""
    assert _same(r, c)


def test_text_is_byte_exact():
    c = copy.deepcopy(CONTENT)
    c["conclusion"]["narrative"] = "총평 글 "
    assert not _same(RESULT, c)
    c["conclusion"]["narrative"] = "총평　글"
    assert not _same(RESULT, c)
    import unicodedata
    c["conclusion"]["narrative"] = unicodedata.normalize("NFD", "총평 글")  # 한글 자모 분해 — 다른 바이트
    assert not _same(RESULT, c)


@pytest.mark.parametrize("broken", [
    {}, {"interpretation": "글"}, {**CONTENT, "recommendations": None}, {**CONTENT, "recommendations": ["x"]},
    {**CONTENT, "conclusion": None},
])
def test_broken_content_never_matches(broken):
    assert not _same(RESULT, broken)


def test_empty_result_does_not_match_empty_content():
    """둘 다 비어 있어도 같은 지문이 되지 않는다(서술 없는 content 가 서술 없는 작업을 통과하지 않게)."""
    assert digest.of_result({}) != digest.of_content({})


# ── J10 검토 ⑧: 자바스크립트 trim 과 같은 글자 집합, 짝 없는 대리 글자 ──


@pytest.mark.parametrize("pid, front", [
    ("﻿P1﻿", "P1"),            # JS trim 은 BOM 을 뗀다(파이썬 strip 은 안 뗌)
    ("　P1 ", "P1"),
    ("\x1cP1\x1f", "\x1cP1\x1f"),        # 파이썬 strip 은 떼지만 JS trim 은 안 뗀다
    ("\x85P1", "\x85P1"),
])
def test_problem_id_trim_uses_the_javascript_set(pid, front):
    r = copy.deepcopy(RESULT)
    r["recommendations"][0]["problem_id"] = pid
    c = copy.deepcopy(CONTENT)
    c["recommendations"][0]["problemId"] = front
    assert _same(r, c)


def test_javascript_trim_set_matches_ecma262():
    ws = {"\t", "\v", "\f", " ", " ", "﻿"} | {chr(c) for c in range(0x10000) if chr(c).isspace()
                                                        and __import__("unicodedata").category(chr(c)) == "Zs"}
    lt = {"\n", "\r", " ", " "}
    assert set(digest.JS_TRIM) == ws | lt


def test_lone_surrogate_gives_a_digest_not_an_exception():
    c = copy.deepcopy(CONTENT)
    c["conclusion"]["narrative"] = "총평 \ud800 글"
    got = digest.of_content(c)
    assert len(got) == 64 and got != digest.of_result(RESULT)
