"""test_session_c_final_review.py — 서술 라운드 세션 C 마지막 검토(c_final_review) 반영.

- 오류 성격 재료: '주된 성격으로 해석해도 된다'는 우세한 쪽 하나가 과반일 때만. 두 오류 합만 과반이면 합이 절반을 넘는다고만 쓴다.
- 8.1 모델 값의 늘린 표기가 숫자 허용 목록에 들어 규칙 문장이 자기 검문을 통과한다.
- 7.2 소수 클래스 크기: 작음 / 작지 않음 / 모름 세 상태 — 모르면 표본 크기를 단정하지 않는다.
- 조사: 프론트(koreanJosa.josaWaGwa)와 같은 공통 규칙 — 같은 대조 표를 두 저장소 테스트에 박는다.
- 8.1 모델 값 표기: 최대 여섯째 자리(프론트 formatBenchmarkValue 와 같은 규칙) — 같은 대조 표를 두 저장소 테스트에 박는다.
"""
import json
from pathlib import Path

import pytest

from app.narrative import generator, prompt, required_items
from app.narrative.baselines import build_benchmark_refs
from app.narrative.derived import compute_derived
from app.narrative.grounding import build_number_whitelist, verify_grounding
from app.narrative.required_items import select_items
from app.narrative.schemas import ErrorRatiosFact, FactSheet, MetricFact, NarrativeRequest
from app.narrative.wording import josa, range_value_text

CONTRACT = Path(__file__).parent / "contract"


def _raw(name: str) -> dict:
    return json.loads((CONTRACT / f"fact_sheet_{name}.json").read_text())


# ── 1. 오류 성격 — '주된 성격'은 우세한 쪽 하나가 과반일 때만 ─────────────────────────────────


def _er(m, b, c, *, threshold=0.5):
    return ErrorRatiosFact(
        unit="sample", misclassified=m,
        boundary={"count": b, "ratio": round(b / m, 4)},
        confident={"count": c, "ratio": round(c / m, 4)},
        threshold=threshold,
    )


def _nature(er: ErrorRatiosFact, task="binary"):
    raw = _raw(task)
    raw["fact_sheet"]["error_ratios"] = json.loads(er.model_dump_json())
    req = NarrativeRequest.model_validate(raw)
    return next(c for c in select_items(req.fact_sheet, None, task) if c.item_id == "7.1-11")


MAIN = "오류의 주된 성격으로 해석해도 된다"
BAN = "오류 전체의 성격으로 일반화하지 않는다"


def test_sum_only_majority_does_not_say_the_dominant_kind_is_over_half():
    # 검토 confirmed #0: 오분류 20건 중 경계 6·확신 5 — 합 11건만 과반인데 '경계 오류가 오분류의 절반을 넘으므로'라고 썼다.
    text = _nature(_er(20, 6, 5)).text
    assert MAIN not in text
    assert "경계 오류가 오분류의 절반을 넘으므로" not in text
    assert "건수가 충분하고 경계 오류와 확신 오류를 합치면 오분류의 절반을 넘는다." in text
    assert "경계 오류 하나만으로는 오분류의 절반을 넘지 않으므로 경계 오류가 오분류의 절반을 넘는다고 쓰지 않는다." in text
    # 일반화 금지 판단(guarded)은 그대로 — 합이 과반이고 건수가 충분하면 막지 않는다. 우세한 오류는 말한다.
    assert BAN not in text and "(우세한 오류는 경계 오류)" in text


def test_dominant_kind_alone_over_half_may_be_read_as_the_main_nature():
    text = _nature(_er(20, 11, 2)).text
    assert "건수가 충분하고 경계 오류가 오분류의 절반을 넘으므로 오류의 주된 성격으로 해석해도 된다." in text
    assert "합치면" not in text and BAN not in text
    # 세션 C 재확인 digits_weak — 249건 중 경계 171건.
    text = _nature(_er(249, 171, 20, threshold=None), "multiclass").text
    assert "건수가 충분하고 경계 오류가 오분류의 절반을 넘으므로 오류의 주된 성격으로 해석해도 된다." in text


def test_few_errors_stay_guarded():
    text = _nature(_er(3, 1, 0)).text
    assert MAIN not in text and "합치면" not in text and BAN in text


def test_overlapping_kinds_do_not_claim_the_sum_is_under_half():
    # 두 오류가 겹칠 수 있으면(임계값 0.85) 합은 부풀 수 있다 — '합쳐도 절반에 이르지 않는다'도 사실이 아닐 수 있다.
    text = _nature(_er(20, 6, 7, threshold=0.85)).text
    assert "합쳐도" not in text and "많은 쪽 오류도 오분류의 절반을 넘지 않아" in text and BAN in text


# ── 2. 8.1 모델 값의 늘린 표기가 허용 목록에 있다 ─────────────────────────────────────────────


def test_benchmark_rule_sentence_passes_its_own_check_with_extended_digits():
    # 0.7999612 의 :g 표기는 '0.799961', 늘린 표기는 '0.79996' — 종전 허용 목록에는 후자가 없었다(검토 confirmed #1).
    fs = FactSheet(verdict="PASS", score=100.0, metrics=[
        MetricFact(metric_id="M2", display_name="Precision", value=0.7999612, status="info"),
        MetricFact(metric_id="M1", display_name="Accuracy", value=0.849961, threshold=0.8, status="pass", value_text="0.850",
                   margin=0.049961, margin_text="0.05", higher_is_better=True),
    ])
    refs = build_benchmark_refs("binary", fs.metrics)
    text = next(c.template_text for c in select_items(fs, None, "binary", benchmark_refs=refs) if c.item_id == "8.1-01")
    assert "Precision(0.79996, 참조 범위 0.8~0.92)" in text and "Accuracy(0.84996, 참조 범위 0.85~0.92)" in text
    result = verify_grounding([text], build_number_whitelist(fs, refs, compute_derived(fs, task_type="binary")))
    assert result.passed, result


# 판정 대상 지표의 목표 판정이 실제 경로(FactSheet → _target_of → benchmark_value_text → 규칙 문장·사실 시트 보기·허용 목록)로
# 8.1 표기에 닿는지(재검토 P2 — 대조 표는 range_value_text 에 target 을 직접 넣어, 연결부가 빠져도 잡지 못했다).
# (값, 목표, margin, 6.1 표기, 기대 8.1 표기, 목표 연결이 없을 때 나올 표기)
TARGET_PATH_CASES = [
    # 잡음만큼 목표 아래지만 프론트가 충족(margin 0)으로 정함 — 위치(아래)와 목표(충족)를 둘 다 맞추는 표기가 없어 목표 우선.
    (0.84999999999, 0.85, 0.0, "0.850", "0.850", "0.849999"),
    # margin 이 없는 옛 요청 — meets_target 의 10자리 반올림으로 충족을 정한다(반올림이 없으면 미달로 보고 '0.849999').
    (0.84999999999, 0.85, None, "0.850", "0.850", "0.849999"),
    # 목표 미달(margin 음수) — 6.1 표기 '0.920' 은 범위 안으로 읽혀 늘린다. 미달을 충족으로 잘못 넘기면 맞는 표기가 없어 '0.920'.
    (0.92004, 0.95, -0.02996, "0.920", "0.92004", "0.92004"),
]


@pytest.mark.parametrize("value, threshold, margin, value_text, expected, _untargeted", TARGET_PATH_CASES)
def test_benchmark_value_uses_target_judgment_through_fact_sheet(value, threshold, margin, value_text, expected, _untargeted):
    from app.narrative import fact_view
    fs = FactSheet(verdict="PASS", score=100.0, metrics=[
        MetricFact(metric_id="M1", display_name="Accuracy", value=value, threshold=threshold,
                   status="pass" if margin is None or margin >= 0 else "fail", value_text=value_text,
                   margin=margin, higher_is_better=True),
    ])
    refs = build_benchmark_refs("binary", fs.metrics)
    text = next(c.template_text for c in select_items(fs, None, "binary", benchmark_refs=refs) if c.item_id == "8.1-01")
    assert f"Accuracy({expected}, 참조 범위 0.85~0.92)" in text, text
    view = fact_view._benchmarks(refs, fs)
    assert [r["모델 값"] for r in view["지표별"] if r["지표"] == "Accuracy"] == [expected]
    result = verify_grounding([text], build_number_whitelist(fs, refs, compute_derived(fs, task_type="binary")))
    assert result.passed, result


def test_target_path_cases_differ_without_target():
    # 위 사례들이 목표 연결을 실제로 시험하는지 — 목표 없이 고르면 처음 두 행은 다른 표기가 나온다.
    for value, _t, _m, value_text, expected, untargeted in TARGET_PATH_CASES[:2]:
        assert range_value_text(value, 0.85, 0.92, value_text) == untargeted != expected


# ── 3. 7.2 소수 클래스 크기 — 작음 / 작지 않음 / 모름 ─────────────────────────────────────────


def _affected_raw(*, small=False, omitted=None, drop_facts=False, facts_only_omitted=False) -> dict:
    raw = _raw("binary")
    fs = raw["fact_sheet"]
    fs["imbalance_effect"] = {
        "level": "affected", "imbalance_ratio": 2.9683, "accuracy_minus_macro_f1": 0.1533,
        "majority_minus_minority_f1": 0.1271, "majority_class": "0", "minority_class": "1",
    }
    fs["facts"]["low_support_classes"] = (
        [{"label": "1", "support": 12, "share": 0.04, "share_text": "0.04"}] if small else []
    )
    if omitted:
        fs["facts"]["omitted"] = omitted
    if drop_facts:
        fs.pop("facts")
    if facts_only_omitted:  # jobs._drop_facts_keep_omitted — facts 형식 오류로 omitted 만 남긴 요청
        fs["facts"] = {"omitted": {"per_class": 1}}
    return raw


def _notes_and_template(raw):
    req = NarrativeRequest.model_validate(raw)
    derived, refs, _, conclusions, ctx, _, _ = generator._prepare(req)
    from app.narrative import fallback
    parts = fallback.section_template("7.2", req.fact_sheet, refs, derived, task_type="binary", conclusions=conclusions)
    return req.fact_sheet, " ".join(ctx.briefs["7.2"].notes), " ".join(v for v in parts.values() if isinstance(v, str))


UNKNOWN_NOTE = ("소수 클래스 1의 표본이 적은지는 이 사실 시트로 확인할 수 없으므로 표본이 적다·적지 않다고 단정하지 않는다. "
                "소수 쪽 성능 지표 해석에 유의해야 한다는 뜻만 적는다.")
NOT_SMALL_NOTE = "소수 클래스 1은 표본이 적은 클래스가 아니므로 표본이 적어서 수치가 흔들린다고 쓰지 않는다"
SMALL_NOTE = "소수 클래스는 표본이 적어 수치가 흔들리기 쉬우니 해석에 유의해야 한다는 뜻으로 적는다"
GENERAL = "소수 클래스의 성능 지표 해석에 유의가 필요하다."


def test_minority_small():
    fs, notes, template = _notes_and_template(_affected_raw(small=True))
    assert required_items.minority_size(fs) == "small"
    assert SMALL_NOTE in notes and NOT_SMALL_NOTE not in notes and "확인할 수 없으므로" not in notes
    assert "표본이 적어 수치가 흔들리기 쉬우므로" in template
    # 목록이 잘렸어도(omitted) 실린 목록에 있으면 작다.
    fs, notes, _ = _notes_and_template(_affected_raw(small=True, omitted={"low_support_classes": 3}))
    assert required_items.minority_size(fs) == "small" and SMALL_NOTE in notes


def test_minority_not_small():
    fs, notes, template = _notes_and_template(_affected_raw())
    assert required_items.minority_size(fs) == "not_small"
    assert NOT_SMALL_NOTE in notes and SMALL_NOTE not in notes
    assert GENERAL in template


@pytest.mark.parametrize("kw", [
    {"drop_facts": True},                                   # 옛 요청 — facts 없음
    {"omitted": {"low_support_classes": 3}},                # 프론트가 표본이 작은 클래스 목록을 잘랐다
    {"omitted": {"facts_class_lists": 7}},                  # 프론트·서버가 서술용 클래스 목록을 통째로 비웠다
    {"facts_only_omitted": True},                           # facts 형식 오류로 omitted 만 남았다(목록을 보내지 않음)
])
def test_minority_unknown(kw):
    fs, notes, template = _notes_and_template(_affected_raw(**kw))
    assert required_items.minority_size(fs) == "unknown"
    assert UNKNOWN_NOTE in notes
    assert NOT_SMALL_NOTE not in notes and SMALL_NOTE not in notes
    assert GENERAL in template and "표본이 적어 수치가 흔들리기 쉬우므로" not in template


# ── 4. 조사 — 두 저장소 공통 대조 표(프론트 koreanJosa.test 와 같은 표) ─────────────────────────

# (이름, 와/과). 규칙: 끝 따옴표·공백 건너뜀 → '이름(설명)'은 괄호 앞 이름 → 한글 받침 → 숫자는 읽는 소리 →
# 영문 대문자 끝은 알파벳 이름(L·M·N·R 만 받침) → 영문 소문자 끝은 -ng·-m·-n·-l·-ck, 모음 뒤 -t 만 받침.
WA_GWA_TABLE = [
    ("benign", "과"), ("malignant", "와"), ("cat", "과"), ("dog", "와"), ("black", "과"), ("test", "와"),
    ("cancer", "와"), ("cell", "과"), ("bus", "와"), ("Class12", "와"), ("P3", "과"), ("2(정상)", "와"),
    ("A(정상)", "와"), ("0(정상)", "과"), ("정상", "과"), ("세포 운명", "과"), ("'benign'", "과"), ("Class1", "과"),
    # 프론트 koreanJosa.test 의 COMMON_WA_GWA 와 같은 두 줄 — % 는 받침 없음, 대문자 끝은 앞 글자와 상관없이 알파벳 이름.
    ("5%", "와"), ("xR", "과"),
]
EUN_NEUN_TABLE = [("cat", "은"), ("P3", "은"), ("Class12", "는"), ("2(정상)", "는")]


@pytest.mark.parametrize("word, expected", WA_GWA_TABLE)
def test_shared_josa_table_wa_gwa(word, expected):
    assert josa(word, "과/와") == expected


@pytest.mark.parametrize("word, expected", EUN_NEUN_TABLE)
def test_shared_josa_table_eun_neun(word, expected):
    assert josa(word, "은/는") == expected


@pytest.mark.parametrize("word, pair, expected", [
    # 정수는 자리 이름까지 읽는다(프론트 numberFinalSound 와 같다): 10 십·100 백·1000 천·10000 만, 10¹² 조(받침 없음).
    ("Class10", "과/와", "과"), ("1000000000000", "과/와", "와"), ("12,000", "은/는", "은"), ("0.5", "과/와", "와"),
    # ㄹ 받침은 '으로/로'에서만 '로'.
    ("cell", "으로/로", "로"), ("URL", "으로/로", "로"), ("cat", "으로/로", "으로"), ("black", "으로/로", "으로"),
    ("Class1", "으로/로", "로"), ("P3", "으로/로", "으로"),
    # 대문자로 끝나면 알파벳 이름(약어).
    ("SVM", "이/가", "이"), ("MCC", "이/가", "가"), ("classL", "은/는", "은"),
    # 정할 수 없으면(기호만·빈 말) 두 꼴을 함께.
    ("", "과/와", "와(과)"), ("—", "은/는", "은(는)"), ("★", "이/가", "이(가)"), ("#", "을/를", "을(를)"),
    ("()", "으로/로", "(으)로"),
])
def test_josa_rule_details(word, pair, expected):
    assert josa(word, pair) == expected


def test_system_prompt_josa_examples_follow_the_shared_rule():
    # 지시문 규칙 8 의 예시(benign과, cat과, dog와, P3이)가 코드의 조사 규칙과 같다.
    for word, pair, particle in (("benign", "과/와", "과"), ("cat", "과/와", "과"), ("dog", "과/와", "와"), ("P3", "이/가", "이")):
        assert f"{word}{particle}" in prompt.SYSTEM_PROMPT
        assert josa(word, pair) == particle, word


# ── 5. 8.1 모델 값 표기 — 두 저장소 공통 대조 표(프론트 judgedNumber.test 와 같은 표) ─────────────

# (값, 범위 하한, 상한, 목표(값, 높을수록 좋음, 충족) 또는 None, 평소 표기(판정 대상의 6.1 표기 — 프론트 formatJudgedValue)
#  또는 None, 기대 문자열). 최대 여섯째 자리, 그래도 위치와 반대면 위치 쪽으로 자른다. 판정 대상이면 같은 자릿수에서
#  목표 판정도 맞아야 하고, 둘 다 맞는 자릿수가 없으면 목표 판정이 우선(평소 표기).
RANGE_TABLE = [
    (0.79996, 0.8, 0.92, None, None, "0.79996"),
    (0.8497, 0.85, 0.92, None, None, "0.8497"),
    (0.92049, 0.85, 0.92, None, None, "0.9205"),
    (0.04996, 0.05, 0.1, (0.05, False, True), "0.050", "0.04996"),      # 낮을수록 좋음, 목표 충족
    (0.10004, 0.05, 0.1, (0.1, False, False), "0.10004", "0.10004"),    # 낮을수록 좋음, 목표 미달
    (0.7999612, 0.8, 0.92, None, None, "0.79996"),
    (0.84999996, 0.85, 0.92, None, None, "0.849999"),                  # 여섯째 자리로도 0.85 — 위치 쪽(내림)으로 자른다
    (0.849961, 0.85, 0.92, (0.85, True, False), "0.84996", "0.84996"),  # 목표 0.85 미달 · 범위 아래
    (0.8499999, 0.85, 0.92, (0.8, True, True), "0.850", "0.849999"),    # 자른 값이 위치·목표 둘 다 맞는다
    (0.84999999999, 0.85, 0.92, (0.85, True, True), "0.850", "0.850"),  # 잡음을 걷으면 목표 충족 — 둘 다 맞는 표기가 없어 목표 우선
]


@pytest.mark.parametrize("value, low, high, target, base, expected", RANGE_TABLE)
def test_shared_range_value_table(value, low, high, target, base, expected):
    assert range_value_text(value, low, high, base, target) == expected
