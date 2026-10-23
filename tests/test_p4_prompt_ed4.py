"""tests/test_p4_prompt_ed4.py — 판 4 지시문(판 4 라운드 S2 ⑤, 대응표 22·23·24·26·29·30, 부록 A4·A11).

- system: 판 4 는 SYSTEM_PROMPT_ED4 — 절 번호 예시·판정 규칙의 자리(3.4절), 통과율 없이 충족 수만, 판정 여유 칸, 평가 말 금지,
  '사용자' 대신 의뢰자. 판 3 규칙 가운데 바꾼 줄 말고는 모두 판 4 에도 글자 그대로 남는다.
- user: 칸을 판 4 자리와 제목으로 부르고(5.1 판정과 불확실성 …, 6절 권고표), '성적서가 따로 찍는 문장'의 자리도 판 4 자리다.
- 8.3-02(판정 여유 작음): 판 4 프론트는 고정 문구 18 을 보내지 않는다 — 판 4 는 4.3 표를 가리키는 해석 지시로 가고, 옛 요청
  갈래('근소 차이' 칸)로 빠지지 않는다. 판 3 쌍둥이는 그 갈래로 간다(갈래가 정말 갈린다).
- D49 ③: 위험 쪽 지표가 있으면 임계값을 말할 수 있는 칸(5.6·5.1·5.5·6)에 그 임계값의 Recall/Precision 과 목표 충족을 함께
  쓰라는 유의. D49 ②: 최약 클래스 결론이 조건(30건·5%·0.1)에 걸릴 때만 운영 위험으로 쓰라고 한다.
- 지시문이 보여 주는 숫자는 모두 판 4 허용 목록 안(자기 검문).
- 배선: 판 4 요청이면 생성기가 실제로 SYSTEM_PROMPT_ED4 를 보낸다(가짜 클라이언트로 본다).

판 3 지시문이 한 글자도 바뀌지 않은 것은 얼린 정답(tests/golden_frozen/edition3/*.job_prompts.json)이 본다.
"""
from __future__ import annotations

import copy
import re

import pytest

from app.narrative import fact_view_ed4, generator, prompt
from app.narrative.grounding import _NUMBER_RE, _canon, strip_context
from app.narrative.jobs import parse_request
from app.narrative.sections import CHAPTERS
from tests.conftest import requested_fields
from tests.p4_example_ed4 import as_edition, edition4_cases, example_raw

CASES = edition4_cases()


def _ctx(raw: dict):
    req, _ = parse_request(copy.deepcopy(raw))
    derived, refs, whitelist, conclusions, ctx, exempt, masks = generator._prepare(req)
    return req, ctx, whitelist


def _chapters(req):
    return [CHAPTERS["7"], CHAPTERS["8"]] + ([CHAPTERS["9"]] if req.fact_sheet.problems else [])


def _strip_same_source(raw: dict) -> dict:
    out = copy.deepcopy(raw)
    out["fact_sheet"]["problems"] = [p for p in out["fact_sheet"]["problems"] if p["kind"] != "same_source"]
    return out


# ── system ──────────────────────────────────────────────────────────────────────────


def _rules(text: str) -> list[str]:
    return [re.sub(r"^\d+\. ", "", line) for line in text.splitlines() if re.match(r"^\d+\. ", line)]


# 판 3 규칙 가운데 판 4 에서 고친 줄(앞부분으로 찾는다) — 나머지는 판 4 에 글자 그대로 있어야 한다.
CHANGED_RULES = (
    "숫자는 사실 시트·결론 목록·고객 입력에 적힌 것만",  # 자릿수를 늘리거나 줄이지 않는다(D49 ④)를 덧붙임
    "성적서의 다른 부분을 가리킬 때는",  # '8.3절' → '5.5절'(A4)
    "판정 규칙은 다음 순서다.",  # 6.1절 표 → 3.4절(A4)
    "통과율과 '몇 개 중 몇 개 충족'은",  # 충족 수만(A11)
    "과장 없이 객관적이고",  # 의뢰자를 '사용자'라고 부르지 않는다(A2)
)


def test_system_prompt_by_edition():
    assert prompt.system_prompt(3) is prompt.SYSTEM_PROMPT
    assert prompt.system_prompt(4) is prompt.SYSTEM_PROMPT_ED4
    with pytest.raises(ValueError):
        prompt.system_prompt(5)


def test_edition4_system_keeps_every_unchanged_edition3_rule():
    rules3, rules4 = _rules(prompt.SYSTEM_PROMPT), _rules(prompt.SYSTEM_PROMPT_ED4)
    assert len(rules3) == 22 and len(rules4) == 24
    changed = [r for r in rules3 if r.startswith(CHANGED_RULES)]
    assert len(changed) == len(CHANGED_RULES)
    for rule in rules3:
        if rule not in changed:
            assert rule in rules4, rule
    for rule in changed:
        assert rule not in rules4, rule
    # 규칙 번호는 판 4 안에서 1부터 차례로.
    numbers = [int(m) for m in re.findall(r"(?m)^(\d+)\. ", prompt.SYSTEM_PROMPT_ED4)]
    assert numbers == list(range(1, 25))


def test_edition4_system_rules():
    s = prompt.SYSTEM_PROMPT_ED4
    for must in ("'5.5절'처럼", "성적서 3.4절에 따로 찍히므로", "통과율(충족한 비율)은 쓰지 않는다", "참고 수치일 뿐 판정의 근거가 아니다",
                 "'판정 여유' 칸('작음' 또는 '충분')", "'아슬아슬'이라는 말은 쓰지 않는다", "'우수'·'미흡'",
                 "의뢰자를 '사용자'라고 부르지 않는다", "자릿수를 늘리거나 줄이지 않는다", "최종 합격", "판정 불가"):
        assert must in s, must
    for banned in ("6.1절 표", "'8.3절'", "추론", "생각 과정", "reasoning", "FAIL", "PASS", "{"):
        assert banned not in s, banned
    # 규칙 번호를 빼면 숫자는 표준 이름(문맥 토큰)·절 번호 예시('5.5절'·'3.4절')·클래스 예시 '1(불량)'뿐이다.
    body = re.sub(r"(?m)^\d+\. ", "", s)
    assert _NUMBER_RE.findall(strip_context(body, edition=4)) == ["1"]
    # 판정 여유 칸 이름은 판 4 사실 보기의 지표 표 칸 이름과 같은 말로 시작한다.
    assert fact_view_ed4.MARGIN_KEY.startswith("판정 여유(")


# ── user: 판 4 자리 ──────────────────────────────────────────────────────────────────


def test_edition4_user_messages_name_edition4_places():
    req, ctx, _ = _ctx(example_raw())
    msg7, msg8, msg9 = (prompt.build_user_message(ctx, ids) for ids in _chapters(req))
    assert "이번 호출은 5.2 오분류, 5.3 데이터 분포, 5.6 추가 관찰 칸을 쓴다." in msg7
    assert "section_7_1 = 5.2 오분류, section_7_2 = 5.3 데이터 분포, section_7_3 = 5.6 추가 관찰." in msg7
    assert "이번 호출은 5.4 참조 범위, 5.1 판정과 불확실성, 5.5 운영 위험 칸을 쓴다." in msg8
    assert "section_9_1 = 6.1 데이터 권고 글, section_9_2 = 6.2 모델과 운영 권고 글, recommendations = 6절 권고표." in msg9
    for head in ("■ 5.2 오분류\n", "■ 5.3 데이터 분포\n", "■ 5.6 추가 관찰\n"):
        assert head in msg7
    for head in ("■ 5.4 참조 범위\n", "■ 5.1 판정과 불확실성\n", "■ 5.5 운영 위험\n"):
        assert head in msg8
    assert "■ 6 개선 권고\n" in msg9 and "6.1 글은 권고 절이 6.1절인" in msg9
    # 다른 칸의 결론·다른 칸에 찍히는 문장도 판 4 자리로.
    assert "  - 5.1 종합 판정은 '최종 합격'이다." in msg7
    assert '  - 4.4절: "확률이 있는 오분류' in msg8
    assert "성적서가 4.4절에 아래 문장으로 찍는다" in msg7
    assert "판정 규칙 문장은 성적서 3.4절에 찍힌다." in msg8
    assert "문제 목록 표(번호·우선순위·문제 이름)는 성적서가 6절에 찍는다." in msg9
    assert "오분류 칸에서" in msg7 and "혼동행렬 해석 칸에서" not in msg7  # D24 선언 단락(번호 없이 판 4 이름)
    # 판 3 자리·판 3 말은 없다.
    for text in (msg7, msg8, msg9):
        for old in ("■ 7.1", "■ 8.2", "■ 9 ", "6.1절 표", "9절 권고표", "7.1절 앞", "8.1절 표 아래", "근소 차이(목표값의 3% 미만)",
                    "혼동행렬 해석 칸", "통과율은 판정 근거가 아니다", "좋은 방향을 반영한 평가"):
            assert old not in text, old


def test_edition3_twin_keeps_edition3_places():
    req, ctx, _ = _ctx(as_edition(_strip_same_source(example_raw()), 3))
    msg8 = prompt.build_user_message(ctx, CHAPTERS["8"])
    assert "■ 8.2 종합 총평\n" in msg8 and "6.1절 표에 찍힌다" in msg8 and "좋은 방향을 반영한 평가" in msg8


def test_edition4_reference_range_task_has_no_value_words():
    task = prompt.section_task("8.1", edition=4)
    assert "참조 범위보다 높음·참조 범위 안·참조 범위보다 낮음" in task and "자체 설정 기준" in task
    assert "좋은 방향을 반영한 평가" not in task and "좋은 방향을 반영한 평가" in prompt.section_task("8.1", edition=3)
    assert prompt.SECTION_TASKS_ED4["8.2"] == prompt.SECTION_TASKS["8.2"]  # 바꾸지 않은 칸은 판 3 글 그대로


# ── 8.3-02: 18 이 없어도 판 4 갈래 ───────────────────────────────────────────────────


def test_judgement_margin_goes_to_interpret_only_without_sentence_18():
    req, ctx, _ = _ctx(example_raw())
    brief = ctx.briefs["8.3"]
    assert any(t.startswith(prompt._INTERPRET_ONLY_ED4["8.3-02"]) for t in brief.interpret)
    assert not any(prompt._WRITE_IT_YOURSELF["8.3-02"] in t for t in brief.must)
    assert any("'판정 여유' 칸이 '작음'인 지표에만" in n for n in brief.notes)
    assert not any("'근소 차이' 칸" in n for n in brief.notes)
    # 판 3 쌍둥이(18 없음)는 옛 요청 갈래 — 갈래가 판으로 정말 갈린다.
    _, ctx3, _ = _ctx(as_edition(_strip_same_source(example_raw()), 3))
    assert any(prompt._WRITE_IT_YOURSELF["8.3-02"] in t for t in ctx3.briefs["8.3"].must)


# ── D49 ③ 위험 쪽 지표 유의 ─────────────────────────────────────────────────────────

RISK_NOTE = "임계값 조정을 말할 때는 의뢰자가 더 위험하다고 밝힌 거짓 음성(FN, 놓침) 쪽 지표인 Recall이"


def test_d49_3_risk_side_note_where_the_threshold_may_be_mentioned():
    _, ctx, _ = _ctx(example_raw())
    with_note = {sid for sid, b in ctx.briefs.items() if any(n.startswith(RISK_NOTE) for n in b.notes)}
    assert with_note == {"7.3", "8.2", "8.3", "9"}
    note = next(n for n in ctx.briefs["7.3"].notes if n.startswith(RISK_NOTE))
    assert "목표를 충족하는지" in note
    raw = example_raw()
    raw["fact_sheet"]["f1_threshold_gain"]["risk_side"]["meets_target_at_best"] = None
    _, ctx_none, _ = _ctx(raw)
    note = next(n for n in ctx_none.briefs["7.3"].notes if n.startswith(RISK_NOTE))
    assert "목표 충족은 쓰지 않는다" in note
    raw["fact_sheet"]["f1_threshold_gain"]["risk_side"] = None
    _, ctx_no, _ = _ctx(raw)
    assert not any("의뢰자가 더 위험하다고 밝힌" in n for b in ctx_no.briefs.values() for n in b.notes)
    _, ctx3, _ = _ctx(as_edition(_strip_same_source(example_raw()), 3))
    assert not any("의뢰자가 더 위험하다고 밝힌" in n for b in ctx3.briefs.values() for n in b.notes)


def test_d49_2_weakest_class_condition_reaches_the_ai():
    req, ctx, _ = _ctx(example_raw())
    msg8 = prompt.build_user_message(ctx, CHAPTERS["8"])
    assert "클래스 사이의 성능 차이를 운영 위험으로 쓰지 않는다." in msg8
    raw = example_raw()
    raw["fact_sheet"]["facts"]["weakest_class"]["low_support"] = True
    _, ctx_risk, _ = _ctx(raw)
    msg8 = prompt.build_user_message(ctx_risk, CHAPTERS["8"])
    assert "이어서 운영 위험으로 다룬다." in msg8 and "운영 위험으로 쓰지 않는다" not in msg8


# ── 지시문의 숫자는 모두 판 4 허용 목록 안 ─────────────────────────────────────────────

# 구조용 숫자(칸 머리표·다른 칸 결론 앞 자리 번호·'이번 호출' 줄·6절 하는 일의 '6.1 글은').
_STRUCTURE = re.compile(r"^■ \d(?:\.\d)? |^  - \d(?:\.\d)?(?:절)? |이번 호출은 [^\n]*|6\.1 글은|6\.2 글은", re.M)


@pytest.mark.parametrize("name, raw", CASES, ids=[c[0] for c in CASES])
def test_every_number_shown_to_the_ai_is_allowed_in_edition4(name, raw):
    req, ctx, whitelist = _ctx(raw)
    assert ctx.edition == 4
    for ids in _chapters(req):
        for regenerate in (False, True):
            msg = _STRUCTURE.sub(" ", prompt.build_user_message(ctx, ids, regenerate=regenerate))
            stripped = strip_context(msg, edition=4)
            tokens = [t for t in _NUMBER_RE.findall(stripped) if _canon(t) is not None]
            assert tokens
            bad = sorted({t for t in tokens if _canon(t) not in whitelist})
            assert not bad, (name, ids, bad)


# ── 배선: 생성기가 판 4 system 을 보낸다 ──────────────────────────────────────────────


@pytest.mark.parametrize("edition", [4, 3])
async def test_generator_sends_the_system_prompt_of_the_request_edition(edition, make_section_client):
    raw = example_raw() if edition == 4 else as_edition(_strip_same_source(example_raw()), 3)
    req, _ = parse_request(copy.deepcopy(raw))
    client = make_section_client({})
    await generator.generate_sections(client, req)
    sent = client.beta.messages.create.call_args_list
    assert len(sent) >= 3
    want = prompt.SYSTEM_PROMPT_ED4 if edition == 4 else prompt.SYSTEM_PROMPT
    assert all(c.kwargs["system"] == want for c in sent)
    chapter8 = next(c.kwargs for c in sent if "section_8_2" in requested_fields(c.kwargs))
    head = "■ 5.1 판정과 불확실성\n" if edition == 4 else "■ 8.2 종합 총평\n"
    assert head in chapter8["messages"][0]["content"]


def test_prompt_edition_arguments_have_no_default():
    import inspect

    for fn in (prompt.section_briefs, prompt._section_notes, prompt._d24_lines, prompt.section_task, prompt.field_name):
        param = inspect.signature(fn).parameters["edition"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is inspect.Parameter.empty, fn.__name__
