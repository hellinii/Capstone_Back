"""test_s5_prompt.py — 작업 경로 장별 지시문(app.narrative.prompt)과 AI 가 읽는 사실 시트 보기(app.narrative.fact_view).

- 재-4·D37: AI 에게 영어 필드명·코드값·지표 번호를 보이지 않는다(운영 서술에 "판정은 FAIL이다"가 나왔다).
- 지시문이 보여 주는 숫자는 모두 숫자 허용 목록 안이어야 한다 — AI 가 지시문에서 옮겨 적은 숫자가 검문에 걸리지 않게.
- 결론 목록을 작성 주체마다 다르게 싣는다(반드시 다룰 / 해석만 / 성적서가 따로 찍음), D24 단락, 고객 입력(D34).
"""
import json
import re
from pathlib import Path

import pytest

from app.narrative import fact_view, generator, prompt
from app.narrative.grounding import _CTX_TOKEN_RE, _NUMBER_RE, _canon, _normalize_minus
from app.narrative.schemas import NarrativeRequest
from app.narrative.sections import CHAPTERS

CONTRACT = Path(__file__).parent / "contract"
NAMES = ("binary", "multiclass", "multilabel", "legacy_main")


def _contract(name: str, **changes) -> NarrativeRequest:
    raw = json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text())
    raw.update(changes)
    return NarrativeRequest.model_validate(raw)


def _ctx(req: NarrativeRequest):
    derived, refs, whitelist, conclusions, ctx, exempt = generator._prepare(req)
    return ctx, whitelist, conclusions


def _chapters(req):
    ids = [CHAPTERS["7"], CHAPTERS["8"]]
    return ids + ([CHAPTERS["9"]] if req.fact_sheet.problems else [])


CLIENT = {
    "model": {"모델명": "loan_model_v2", "버전": "1.0", "모델 용도": "대출 심사 보조"},
    "declarations": {"위험 방향": "FN 이 더 위험"},
    "training_data": {"학습 표본 수": "12000"},
    "class_label_descriptions": {"0": "정상", "1": "불량"},
    "company": {"회사명": "비밀주식회사", "사업자번호": "123-45-67890"},
}


# ── 영어 이름을 보이지 않는다 ─────────────────────────────────────────────────────

_SNAKE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")
_FIELD_LINE = re.compile(r"응답 필드와 칸: [^\n]*")


@pytest.mark.parametrize("name", NAMES)
def test_user_messages_have_no_english_field_names_codes_or_metric_ids(name):
    req = _contract(name, client_inputs=CLIENT)
    ctx, _, _ = _ctx(req)
    for ids in _chapters(req):
        msg = prompt.build_user_message(ctx, ids)
        body = _FIELD_LINE.sub("", msg)  # 응답 필드 이름 안내 줄(스키마 키)만 뺀다
        leaked = set(_SNAKE.findall(body)) - {"loan_model_v2"}  # 고객이 준 모델명은 이름이다
        assert not leaked, (name, ids, leaked)
        assert not re.search(r"(?<![A-Za-z_])(?:PASS|FAIL|CONDITIONAL_PASS|INCONCLUSIVE)(?![A-Za-z_])", body), (name, ids)
        assert not re.search(r"(?<![A-Za-z0-9])M\d{1,2}(?![A-Za-z0-9])", body), (name, ids)
        assert "고정 문구" not in body  # 문구 번호를 베끼면 숫자 검문에 걸린다
        for code in ("label_cell", "low_support", "best_threshold", "core_fail", "affected", "realtime", "HIGH"):
            assert code not in body, (name, code)


@pytest.mark.parametrize("name", NAMES)
def test_fact_view_keys_are_korean(name):
    ctx, _, _ = _ctx(_contract(name))

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    ascii_keys = {k for k in keys(ctx.view) if re.fullmatch(r"[A-Za-z_]+", k) and "_" in k}
    assert not ascii_keys


def test_system_prompt_rules():
    s = prompt.SYSTEM_PROMPT
    for must in ("최종 합격", "조건부 합격", "최종 불합격", "판정 불가", "참고 수치", "판정의 근거가 아니다",
                 "이름과 의뢰자가 준 설명", "영어 코드값과 필드명", "서술체의 완결된 문장", "추가 관찰",
                 "모델 구조나 학습 기법의 상세한 처방은 쓰지 않는다", "예측 수치", "'8.3절'처럼"):
        assert must in s, must
    # 추론 과정을 적으라는 요청은 두지 않는다(reasoning_extraction 거절 — 대체 모델이 없다).
    for banned in ("추론", "생각 과정", "근거를 적", "reasoning", "FAIL", "PASS"):
        assert banned not in s, banned
    # 사실과 무관한 고정 규칙이다 — 사실 시트 값이 들어가지 않는다.
    assert "{" not in s


# ── 지시문의 숫자는 모두 허용 목록 안 ──────────────────────────────────────────────

_STRUCTURE = re.compile(
    r"^■ \d(?:\.\d)? |^  - \d(?:\.\d)? |이번 호출은 [^\n]*|9\.1 글은|9\.2 글은", re.M,
)


@pytest.mark.parametrize("name", NAMES)
def test_every_number_shown_to_the_ai_is_allowed(name):
    req = _contract(name, client_inputs=CLIENT)
    ctx, whitelist, _ = _ctx(req)
    for ids in _chapters(req):
        msg = _STRUCTURE.sub(" ", prompt.build_user_message(ctx, ids))
        stripped = _CTX_TOKEN_RE.sub(" ", _normalize_minus(msg))
        bad = sorted({t for t in _NUMBER_RE.findall(stripped) if _canon(t) is not None and _canon(t) not in whitelist})
        assert not bad, (name, ids, bad)


def test_system_prompt_numbers_are_only_structure():
    # 규칙 번호를 빼면 표준 이름(문맥 토큰으로 지워짐)·절 번호 예시('8.3절')·클래스 예시 '1(불량)'뿐이다.
    body = re.sub(r"(?m)^\d+\. ", "", prompt.SYSTEM_PROMPT)
    assert _NUMBER_RE.findall(_CTX_TOKEN_RE.sub(" ", body)) == ["1"]


# ── 결론 목록 → 칸별 지시 ────────────────────────────────────────────────────────

def test_conclusions_are_placed_by_author():
    req = _contract("binary")
    ctx, _, conclusions = _ctx(req)
    msg7 = prompt.build_user_message(ctx, CHAPTERS["7"])
    for c in conclusions:
        if c.section == "7.1" and c.author in ("ai", "ai_fact"):
            assert c.text in msg7
    # 해석만: 고정 문구 17 의 문장을 보여 주고 숫자를 되풀이하지 말라고 한다.
    seventeen = next(s.text for s in req.fact_sheet.facts.fixed_sentences if s.id == "17")
    assert "해석만 쓸 결론" in msg7 and seventeen in msg7
    # 결론 문자열의 AI 용 설명(번호)은 싣지 않는다.
    assert "고정 문구 17" not in msg7
    msg8 = prompt.build_user_message(ctx, CHAPTERS["8"])
    assert "Accuracy 0.505(목표 0.510 이상, 미달)" in msg8  # 고정 문구 18 — 지표 번호를 뺐다
    assert "Accuracy(M1)" not in msg8
    assert "성적서가 따로 찍는 문장" in msg8 and "6.1절 표에 찍힌다" in msg8


def test_each_call_sees_other_sections_conclusions_as_reference_only():
    req = _contract("binary")
    ctx, _, _ = _ctx(req)
    msg7 = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "[다른 칸의 결론 — 참고만 하고 이번 호출에서는 쓰지 않는다]" in msg7
    assert "8.2 종합 판정은 '최종 불합격'이다." in msg7
    assert "■ 8.2" not in msg7


def test_d24_paragraph_only_in_the_call_that_writes_the_section():
    req = _contract("binary")
    ctx, _, _ = _ctx(req)
    msg7 = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "[의뢰자 선언 — 빠뜨리지 말고 숫자로 반영]" in msg7 and "FN(놓침)이 더 위험하다고 밝혔다" in msg7
    assert "[의뢰자 선언" not in prompt.build_user_message(ctx, CHAPTERS["8"])


def test_regenerate_message_names_only_that_section():
    req = _contract("binary")
    ctx, _, _ = _ctx(req)
    msg = prompt.build_user_message(ctx, ("7.1",), regenerate=True)
    assert "7.1 혼동행렬 해석 칸을 다시 쓴다" in msg and "■ 7.2" not in msg
    assert "section_7_1 = 7.1 혼동행렬 해석" in msg


# ── 사실에 따른 유의점 ──────────────────────────────────────────────────────────

def test_fact_dependent_notes():
    req = _contract("binary")
    ctx, _, _ = _ctx(req)
    b = ctx.briefs
    assert any("경계 오류는 계산하지 않았다" in n for n in b["7.1"].notes)
    assert any("불균형이 경미하다" in n for n in b["7.2"].notes)
    assert any("유의성 검정은 하지 않았다" in n for n in b["8.2"].notes)
    assert any("의뢰자가 지정한 핵심 지표라고 쓰지 않는다" in n for n in b["8.2"].notes)
    assert any("지연시간은 측정하지 않았다" in n for n in b["8.3"].notes)
    ml = _ctx(_contract("multilabel"))[0].briefs
    assert any("레이블 칸" in n for n in ml["7.1"].notes)
    assert any("전체를 넘을 수 있다" in n for n in ml["7.2"].notes)


def test_view_uses_judged_texts_and_korean_values():
    ctx, _, _ = _ctx(_contract("binary"))
    acc = next(m for m in ctx.view["지표"] if m["지표"] == "Accuracy")
    assert (acc["값"], acc["결과"], acc["목표와의 차이(0 이상이면 충족 쪽)"]) == ("0.505", "미달", "-0.005")
    assert ctx.view["판정"]["종합 판정"] == "최종 불합격"
    assert ctx.view["문제 목록"][0]["우선순위"] == "높음" and ctx.view["문제 목록"][0]["권고 절"] == "9.2절"
    assert "신뢰구간 방식" in ctx.view and "M1" not in ctx.view["신뢰구간 방식"]
    assert "Wilson 점수 구간(Accuracy)" in ctx.view["신뢰구간 방식"]


def test_evidence_translation():
    out = fact_view.translate_evidence({
        "cause": "판정 대상 지표 없음", "reason": "no_judged", "metrics": [], "judged_count": 0,
        "level": "affected", "imbalance_ratio": 2.9, "display": {"imbalance_ratio": "2.9"}, "low_support": True,
    })
    assert out == {"원인": "판정 대상 지표 없음", "지표": [], "판정 대상 지표 수": 0,
                   "판단": fact_view.IMBALANCE_KO["affected"], "불균형 비율": "2.9", "표본이 작음": "예"}


def test_name_metric_ids():
    names = {"M1": "Accuracy", "M4": "F1 Score"}
    assert fact_view.name_metric_ids("Accuracy(M1) 0.5, F1 Score(M4) 0.6", names) == "Accuracy 0.5, F1 Score 0.6"
    assert fact_view.name_metric_ids("구간(M1), 부트스트랩(…; M4, M9)", names) == "구간(Accuracy), 부트스트랩(…; F1 Score, M9)"


# ── 고객 입력(D25·D34)·클래스 설명(D37)·덜어 낸 표 ────────────────────────────────

def test_client_view_keeps_known_groups_in_korean_and_drops_the_rest():
    view, dropped = fact_view.build_client_view(CLIENT)
    assert set(view) == {"모델", "의뢰자 선언", "학습 데이터", "클래스 설명"}
    assert dropped == ["company"]
    req = _contract("binary", client_inputs=CLIENT)
    ctx, _, _ = _ctx(req)
    msg = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "비밀주식회사" not in msg and "123-45-67890" not in msg
    assert "[클래스 이름과 설명]" in msg and "0(정상), 1(불량)" in msg
    assert ctx.dropped_client_keys == ["company"]


def test_omitted_tables_are_announced():
    raw = json.loads((CONTRACT / "fact_sheet_multiclass.json").read_text())
    raw["fact_sheet"]["facts"]["omitted"] = {"per_class": 30, "confusion": 30}
    ctx, whitelist, _ = _ctx(NarrativeRequest.model_validate(raw))
    msg = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "[덜어 낸 표]" in msg and "클래스별 성능(30개 뺌)" in msg and "혼동행렬(30개 뺌)" in msg
    assert "없다고 쓰지 않는다" in msg


def test_prompt_trimming_drops_big_tables_first(monkeypatch):
    req = _contract("multiclass")
    ctx0, _, _ = _ctx(req)
    full = len(prompt.SYSTEM_PROMPT) + prompt._longest_message(ctx0)
    monkeypatch.setattr(prompt, "MAX_PROMPT_CHARS", full - 1)
    ctx, whitelist, _ = _ctx(req)
    assert ctx.trimmed[0] == "confusion.matrix"  # 덜어 낸 안내가 붙어 다음 표까지 덜어 낼 수 있다
    assert "행렬(행 = 실제, 열 = 예측)" not in ctx.view.get("혼동행렬", {})
    assert ctx.view["덜어 낸 표(일부만 실림)"]["혼동행렬"] > 0
    assert prompt.fits(ctx)
    # 덜어 낸 수는 서술이 써도 된다(허용 목록에 넣었다).
    assert _canon(str(ctx.view["덜어 낸 표(일부만 실림)"]["혼동행렬"])) in whitelist


def test_numbers_printed_in_other_sections_are_shown_with_a_do_not_repeat_note():
    """S5 실측: 8.3·9.2 가 7.1 의 경계·확신 오류 건수(고정 문구 17)를 되풀이했다 — 다른 장 호출에도 알린다."""
    req = _contract("binary")
    ctx, _, _ = _ctx(req)
    seventeen = next(s.text for s in req.fact_sheet.facts.fixed_sentences if s.id == "17")
    msg8 = prompt.build_user_message(ctx, CHAPTERS["8"])
    assert "[다른 칸에 성적서가 숫자로 찍는 문장" in msg8 and f'7.1절: "{seventeen}"' in msg8
    msg7 = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "8.3절: " in msg7.split("[다른 칸에 성적서가 숫자로 찍는 문장")[1]
    assert f'7.1절: "{seventeen}"' not in msg7  # 자기 칸 문장은 '해석만 쓸 결론'에 있다



# ── S5 독립 검토(wf_a65c97fb-fec) 반영 ─────────────────────────────────────────

def test_evidence_uses_engine_display_texts():
    """prompt-1: 근거의 숫자 칸을 판정과 같은 쪽 표기로 — 0.09996 이 0.1 로 보이지 않게."""
    out = fact_view.translate_evidence({
        "level": "affected", "imbalance_ratio": 2.0, "accuracy_minus_macro_f1": 0.1, "majority_minus_minority_f1": 0.25,
        "display": {"imbalance_ratio": "2", "accuracy_minus_macro_f1": "0.09996", "majority_minus_minority_f1": "0.25"},
    })
    assert out["정확도 − 매크로 F1"] == "0.09996"
    gain = fact_view.translate_evidence({"gain": 0.1, "gain_text": "0.09996", "notable": False})
    assert gain == {"향상폭": "0.09996", "임계값 부적합": "아니오"}


def test_imbalance_ratio_in_distribution_uses_the_display_text():
    """prompt-2: 1.5 와 비교되는 값 — 1.49996 이 1.5 로 실리지 않는다."""
    raw = json.loads((CONTRACT / "fact_sheet_multiclass.json").read_text())
    raw["fact_sheet"]["distribution"]["imbalance_ratio"] = 1.49996
    raw["fact_sheet"]["imbalance_effect"].update(level="mild", imbalance_ratio=1.5)
    raw["fact_sheet"]["imbalance_effect"]["display"]["imbalance_ratio"] = "1.49996"
    ctx, _, _ = _ctx(NarrativeRequest.model_validate(raw))
    assert ctx.view["분포·불균형"]["불균형 비율"] == "1.49996"


def test_trimmed_fixed_sentences_are_named_in_korean_and_interpret_only_falls_back():
    """prompt-3: 프론트가 고정 문장을 덜어 냈으면 영어 키를 흘리지 않고, '아래 문장' 없는 해석만 지시도 하지 않는다."""
    raw = json.loads((CONTRACT / "fact_sheet_binary.json").read_text())
    raw["fact_sheet"]["facts"]["fixed_sentences"] = []
    raw["fact_sheet"]["facts"]["omitted"] = {"fixed_sentences": 3, "mystery_key": 2}
    ctx, _, _ = _ctx(NarrativeRequest.model_validate(raw))
    msg = prompt.build_user_message(ctx, CHAPTERS["7"])
    assert "fixed_sentences" not in msg and "mystery_key" not in msg
    assert "성적서 고정 문장(3개 뺌)" in msg and "그 밖의 표(2개 뺌)" in msg
    assert "아래 문장" not in msg
    # S6: 덜어 낸 것은 AI 에게 보내는 목록뿐이고 화면은 17 을 그대로 찍는다(프론트 assembleEvaluatedReport 의 errorRatioSentence
    # 는 facts 자르기와 무관) — AI 가 숫자를 직접 쓰면 두 번 찍힌다. 그래서 '해석만'을 '따로 숫자로 찍는다'로 알린다.
    assert "성적서가 따로 숫자로 찍는다" in msg
    assert "'오분류의 성격' 값 그대로 쓰고" not in msg
    legacy = _ctx(_contract("legacy_main"))[0]
    legacy_msg = prompt.build_user_message(legacy, CHAPTERS["7"])
    assert "아래 문장" not in legacy_msg
    assert "'오분류의 성격' 값 그대로 쓰고" in legacy_msg  # 옛 요청(facts 없음)은 성적서가 찍지 않는다


def test_regenerating_one_section_does_not_forbid_its_own_chapters_numbers():
    """prompt-4: 8.1 만 다시 만들 때 같은 장의 문구 18 을 '되풀이 금지'로 싣지 않는다."""
    ctx, _, _ = _ctx(_contract("binary"))
    msg = prompt.build_user_message(ctx, ("8.1",), regenerate=True)
    assert "8.3절: " not in msg
    assert "7.1절: " in msg  # 다른 장(7장)의 문구 17 은 그대로 알린다


def test_class_names_that_look_like_metric_ids_are_kept():
    """prompt-5: 클래스가 M1·M2 꼴이면 지표 이름으로 바꾸지 않는다."""
    names = {"M1": "Accuracy", "M2": "Precision"}
    assert fact_view.name_metric_ids("구간(M1), 클래스 M2", names, {"M2"}) == "구간(Accuracy), 클래스 M2"
    raw = json.loads((CONTRACT / "fact_sheet_multiclass.json").read_text())
    raw["fact_sheet"]["problems"][-1]["title"] = "학습↔평가 분포 차이 (M1, M2, M3)"
    ctx, _, _ = _ctx(NarrativeRequest.model_validate(raw))
    assert ctx.view["문제 목록"][-1]["문제"] == "학습↔평가 분포 차이 (M1, M2, M3)"
