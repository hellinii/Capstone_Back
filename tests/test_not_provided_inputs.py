"""고객 입력의 '제공되지 않음 (의뢰자 제공: …)' — 값은 제공 아님, 괄호 글은 의뢰자 참고 글(세션 B 검토 input-1).

프론트 materialRowText 는 '모름 + 글'·'기타 빈칸 + 글'을 "제공되지 않음 (의뢰자 제공: …)" 으로 보낸다(점검 표 2.0·1.4 는
빠짐으로 센다). 백엔드가 고객 입력의 제공 여부를 보는 자리(지표 선택 이유 8.2-12·모델 용도·클래스 설명·지시문 규칙 17)가
모두 '시작하는가'로 같은 기준을 쓰는지 자리마다 따로 고정한다.
"""
import pytest

from app.narrative import prompt
from app.narrative.prompt import SYSTEM_PROMPT
from app.narrative.required_items import select_items
from app.narrative.schemas import FactSheet
from app.narrative.sections import CHAPTERS
from app.narrative.wording import NOT_PROVIDED, class_descriptions, is_not_provided

WITH_NOTE = "제공되지 않음 (의뢰자 제공: 팀에서 정하지 못했음)"


def _fs() -> FactSheet:
    return FactSheet(verdict="PASS", score=33.3, n_samples=7, dropped_rows=3)


def _ids(client: dict, task: str = "multilabel") -> dict[str, str]:
    return {c.item_id: c.text for c in select_items(_fs(), client, task, edition=3)}


@pytest.mark.parametrize("value", [None, "", "  ", 3, NOT_PROVIDED, f" {NOT_PROVIDED} ", WITH_NOTE])
def test_is_not_provided_true(value):
    assert is_not_provided(value)


@pytest.mark.parametrize("value", ["레이블 오류를 고르게 본다", "기타 (의뢰자 제공: 제공되지 않음)"])
def test_is_not_provided_false(value):
    # 값이 있고 글이 붙은 꼴("값 (의뢰자 제공: …)")은 제공이다 — 시작만 본다.
    assert not is_not_provided(value)


def test_metric_rationale_with_note_is_not_provided():
    assert "8.2-12" not in _ids({"metric_rationale": WITH_NOTE})
    # 반대 갈래 — 진짜 이유면 살아난다(같은 입력 모양에서 판정만 갈린다).
    assert "8.2-12" in _ids({"metric_rationale": "레이블 오류를 고르게 본다"})


def _chapter_8_message(client: dict) -> str:
    fs = _fs()
    ctx = prompt.build_context(
        fs, {}, [], task_type="multilabel", report_purpose="", client_inputs=client,
        conclusions=select_items(fs, client, "multilabel", edition=3), edition=3,
    )
    return prompt.build_user_message(ctx, CHAPTERS["8"])


def test_prompt_does_not_say_client_picked_reason_when_not_provided():
    # 반대 갈래 먼저 — 진짜 이유면 지시문의 '성적서가 따로 찍는 문장'에 그 말이 실린다(단정이 이빨을 갖게).
    assert "의뢰자가 고른 이유" in _chapter_8_message({"metric_rationale": "레이블 오류를 고르게 본다"})
    assert "의뢰자가 고른 이유" not in _chapter_8_message({"metric_rationale": WITH_NOTE})


def test_model_purpose_with_note_is_not_provided():
    given = {c.item_id for c in select_items(_fs(), {"model": {"모델 용도": "불량 선별"}}, "binary", edition=3) if "모델 용도" in c.text}
    assert given  # 용도가 있으면 쓰는 항목이 있음을 먼저 단정
    texts = [c.text for c in select_items(_fs(), {"model": {"모델 용도": WITH_NOTE}}, "binary", edition=3)]
    assert not any("모델 용도" in t for t in texts)
    assert not any("팀에서 정하지 못했음" in t for t in texts)


CLASS_DESC = {"class_label_descriptions": {"0": "정상", "1": WITH_NOTE, "2": NOT_PROVIDED}}


def test_class_description_with_note_is_dropped():
    assert class_descriptions(CLASS_DESC) == {"0": "정상"}


def test_prompt_class_descriptions_use_the_same_rule():
    # 지시문의 [클래스 이름과 설명] 도 규칙 문장과 같은 한 벌(wording.class_descriptions)을 쓴다.
    fs = _fs()
    ctx = prompt.build_context(fs, {}, [], task_type="multiclass", report_purpose="", client_inputs=CLASS_DESC,
                               conclusions=select_items(fs, CLASS_DESC, "multiclass", edition=3), edition=3)
    assert ctx.class_descriptions == {"0": "정상"}


def test_prompt_rule_17_treats_note_as_reference_only():
    assert "'제공되지 않음'으로 시작하면" in SYSTEM_PROMPT
    assert "참고 글" in SYSTEM_PROMPT
