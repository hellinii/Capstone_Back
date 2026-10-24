"""tests/test_p4_contract_ed4.py — 판 4 공통 계약(판 4 라운드 S2 ①)의 백엔드 쪽.

공통 파일(같은 바이트 사본이 프론트 src/test/contract/ 에 있다)을 지킨다.
- `fact_sheet_contract_ed4.json`: 판 4 서술 요청 명세(판 신호 report_edition, 판 4 에만 있는 키, 판 4 표기로 찍는 칸).
- `number_wording_ed4.json`: 판 4 숫자 표기·조사 대조 표. 두 저장소가 같은 표로 자기 함수를 시험한다.
- `narrative_slots_ed4.json`: 서술 칸 → 판 4 자리, 판 4 제목, 작성 기록 줄(B25·B26).
- `fact_sheet_{binary,multiclass,multilabel}_ed4.json`: 판 4 예시 — 프론트가 앱이 실제로 보내는 판 4 요청 본문으로 뜬 것
  (S3 ⑧, 프론트 src/test/edition4/contractExamplesEd4.test.tsx). 입력은 판 3 발급본 RPT-2026-0001(bc_strong)·0003(digits_strong)·
  0005(yeast_strong)과 같은 입력에 새 평가 응답. 이진은 S2 ① 이 명세대로 손으로 고친 임시 예시와 바이트까지 같았다(sha 그대로).
  예시를 바꾸면 프론트에서 다시 뜨고(UPDATE_CONTRACT) 이 폴더로 바이트 그대로 옮긴 뒤 두 시험의 sha256 을 이유와 함께 고친다.

판 3 명세·예시(fact_sheet_contract.json, fact_sheet_{binary,multiclass,multilabel,legacy_main}.json)는 프론트가 sha256 으로
묶어 둔 판 3 정답이라 건드리지 않는다. 이 시험은 그 파일들에 판 4 키가 없다는 것(null 키 금지)만 본다.

대조 표는 시험 안의 기준 구현(Decimal ROUND_HALF_UP — 프론트 toFixed 와 같은 반올림)으로 표 자체가 규칙과 맞는지 보고,
조사는 이미 있는 `wording.josa` 로 본다. 판 4 숫자 함수(ed4_plain 등)는 S2 ⑥ 이 만들고 같은 표로 시험한다.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from app.narrative.sections import SECTION_IDS
from app.narrative.wording import josa

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
FRONT_CONTRACT = TESTS.parents[1] / "Capstone_Front" / "src" / "test" / "contract"
ROUND_SPEC = TESTS.parents[1] / "NARRATIVE_ROUND" / "사실시트_계약_ed4.json"
SHARED = ("fact_sheet_contract_ed4.json", "number_wording_ed4.json", "narrative_slots_ed4.json")
EXAMPLE = "fact_sheet_binary_ed4.json"
# 판 4 예시 → 같은 입력의 판 3 발급본 요청(tests/data/edition3_issued/requests — 판 3 정답 입력).
EXAMPLES = {
    EXAMPLE: "bc_strong",
    "fact_sheet_multiclass_ed4.json": "digits_strong",
    "fact_sheet_multilabel_ed4.json": "yeast_strong",
}

# 자기 저장소 안의 고정(옆 저장소가 없는 CI 에서도 지킨다). 프론트 contractEd4.test.ts·contractExamplesEd4.test.tsx 와 같은 값이다.
# S2·S3 검토(2026-10-24)로 고침: 명세 — D1·D2·D3·D4·D6·D7 결정(decided), 판 4 전용 새 키 facts.weakest_class.f1_text, '임시' 글(D8)
# 걷음. 대조 표 — target·latency·distinct 규칙과 줄, given 정수·지수 꼴(T15), judged D1·여섯째 자리(T13) 줄, 조사 줄(T14).
# 예시 셋 — weakest_class 에 f1_text 한 줄(f1 바로 다음: 이진 0.977 = M4 value_text, 다중 0.9720 = 5 와 달라 보이게 늘림, 멀티
# 0.048). 프론트가 이 칸을 만든 뒤 다시 뜬 f1_text 줄은 손 글자와 같았다. 같은 검토 D6(문제 근거 숫자도 판 4 글자 — 지표 근거
# value_text·ed4Target·margin_text, 최약 클래스 근거 f1_text·share_text·gap_text·매크로 F1(D2), 불균형 근거 display 판 4)으로
# 세 예시의 problems[].evidence 가 바뀌어 프론트에서 다시 떠 옮겼다(그 밖 바이트 그대로).
# 후속(2026-10-24)으로 명세·대조 표만 다시 고침: 명세 — 멀티레이블 4.5 학습·평가 비율의 분모 = 표본 수(T3 사용자 답, edition4_meaning
# 두 줄·새 상태 training_samples_not_provided), D3 클래스별 표(decided·function_names.class_f1), not_decided 에 B16 이유 질문.
# 대조 표 — distinct 규칙 글과 class_table 줄 다섯(others = [최약 F1]). 예시 셋·칸 대응표는 그대로.
# 후속 검토(2026-10-24)로 명세만 다시 고침: B33(초안) 사용자 승인 — not_decided 의 B16 이유 질문을 decided 로 옮기고 승인 글자를
# 옮겨 적음, edition4_meaning 의 distribution_shift_status 줄도 같이. 대조 표·칸 대응표·예시 셋은 그대로.
SHA256 = {
    "fact_sheet_contract_ed4.json": "15fb4c3e4d507850c4594ed510cd29b34cbdaeadece9945a6ee1c171f16b5148",
    "number_wording_ed4.json": "6d4659d849363ceac749108c19b9a41662b18ce6e1f02bdc30586f54142466fb",
    "narrative_slots_ed4.json": "3c43ffa35357c8c7f4c8a30d182482a85dec51caefa872dc9e556ae51d5b7d35",
    EXAMPLE: "0f7da6dfc88870aec7cb18c3a02204319df8487b4a9e058c344ade8dc0d63f69",
    "fact_sheet_multiclass_ed4.json": "68277d563888fdc1fdb1f4c50a0cd373d264394026c0ede9e1c421bc91ef96a5",
    "fact_sheet_multilabel_ed4.json": "260783d76d0cc4d876a2c7b22f18d2303977223ca32e55c58bc51c35d03d2d26",
}


def _load(name: str) -> dict:
    return json.loads((CONTRACT / name).read_text(encoding="utf-8"))


SPEC = _load("fact_sheet_contract_ed4.json")
TABLE = _load("number_wording_ed4.json")
SLOTS = _load("narrative_slots_ed4.json")
EX = _load(EXAMPLE)
ED4_ONLY = [entry["path"] for entry in SPEC["edition4_only_keys"]]


# ── 파일 고정 ────────────────────────────────────────────────────────────────────


def test_files_are_pinned_by_sha256():
    actual = {name: hashlib.sha256((CONTRACT / name).read_bytes()).hexdigest() for name in SHA256}
    assert actual == SHA256


@pytest.mark.skipif(not FRONT_CONTRACT.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
@pytest.mark.parametrize("name", SHARED + tuple(EXAMPLES))
def test_frontend_copy_is_byte_identical(name):
    assert (FRONT_CONTRACT / name).read_bytes() == (CONTRACT / name).read_bytes()


@pytest.mark.skipif(not ROUND_SPEC.exists(), reason="라운드 폴더가 없다")
def test_round_folder_spec_is_byte_identical():
    assert ROUND_SPEC.read_bytes() == (CONTRACT / "fact_sheet_contract_ed4.json").read_bytes()


# ── 경로 도구 ────────────────────────────────────────────────────────────────────


def _containers(obj, path: str) -> list[tuple[dict, str]]:
    """'a.b[].c' 의 마지막 키를 담을 dict 들과 그 키. 중간이 없거나 null 이면 그 갈래는 없다."""
    parts = path.split(".")
    nodes = [obj]
    for part in parts[:-1]:
        many = part.endswith("[]")
        key = part[:-2] if many else part
        nxt = []
        for node in nodes:
            if not isinstance(node, dict) or node.get(key) is None:
                continue
            value = node[key]
            if many:
                nxt.extend(v for v in value if isinstance(v, dict))
            else:
                nxt.append(value)
        nodes = nxt
    return [(n, parts[-1]) for n in nodes if isinstance(n, dict)]


def _diff_paths(a, b, path: str = "") -> set[str]:
    """두 JSON 의 다른 자리(목록 번호는 [] 로 모음). 목록 길이가 다르면 그 목록 자리 하나."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: set[str] = set()
        for key in set(a) | set(b):
            here = f"{path}.{key}" if path else key
            if key not in a or key not in b:
                out.add(here)
            else:
                out |= _diff_paths(a[key], b[key], here)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return {path}
        out = set()
        for x, y in zip(a, b):
            out |= _diff_paths(x, y, f"{path}[]")
        return out
    if type(a) is not type(b) or a != b:
        return {path}
    return set()


# ── 판 4 키: 판 4 예시에는 모두, 판 3 요청에는 하나도 ─────────────────────────────


def _edition3_requests() -> dict[str, dict]:
    files = [CONTRACT / f"fact_sheet_{t}.json" for t in ("binary", "multiclass", "multilabel", "legacy_main")]
    files += sorted((TESTS / "data" / "edition3_issued" / "requests").glob("*.json"))
    files += sorted((TESTS / "golden_frozen" / "inputs").glob("fact_sheet_*.json"))
    return {f"{p.parent.name}/{p.name}": json.loads(p.read_text(encoding="utf-8")) for p in files}


def test_edition4_only_keys_are_declared_once():
    assert len(ED4_ONLY) >= 10
    assert len(set(ED4_ONLY)) == len(ED4_ONLY)
    assert ED4_ONLY[0] == "report_edition"
    # facts 안에 더한 판 4 키는 둘뿐이다(명세 원칙 — 최약 클래스 F1 글자는 S2·S3 검토 D2·D3 로 더함).
    assert [p for p in ED4_ONLY if ".facts." in p] == [
        "fact_sheet.facts.distribution_shift[].evaluation_minus_training_text", "fact_sheet.facts.weakest_class.f1_text",
    ]


@pytest.mark.parametrize("path", ED4_ONLY)
def test_edition4_example_has_every_edition4_key(path):
    sites = _containers(EX, path)
    assert sites, path  # 빈 갈래를 돌면 늘 통과한다
    assert all(key in node for node, key in sites), path


def test_edition3_requests_have_no_edition4_keys():
    requests = _edition3_requests()
    assert len(requests) >= 13  # 계약 4 + 발급본 6 + 얼린 입력 6
    hits = [
        f"{name}: {path}"
        for name, req in requests.items()
        for path in ED4_ONLY
        for node, key in _containers(req, path)
        if key in node
    ]
    assert hits == []
    kinds = {p["kind"] for req in requests.values() for p in (req.get("fact_sheet") or {}).get("problems") or []}
    assert kinds and "same_source" not in kinds


def test_example_differs_from_edition3_bc_strong_only_at_declared_places():
    ed3 = json.loads((TESTS / "data" / "edition3_issued" / "requests" / "bc_strong.request.json").read_text(encoding="utf-8"))
    declared = set(ED4_ONLY) | {k for k in SPEC["edition4_notation"] if not k.startswith("$")} | set(SPEC["edition4_meaning"])
    diff = _diff_paths(ed3, EX)
    assert diff
    # 목록 길이가 달라진 자리(problems·fixed_sentences)는 그 아래 칸이 선언돼 있으면 선언된 것으로 본다.
    stray = sorted(p for p in diff if p not in declared and not any(d.startswith(f"{p}[]") or d == p for d in declared))
    assert stray == []
    # 판 4 키는 모두 판 3 에 없던 자리라 diff 에 나온다.
    assert set(ED4_ONLY) <= diff
    # 판 3 와 같은 값으로 남아야 하는 칸 몇 개 — 판 4 가 건드리면 안 된다.
    for path in ("fact_sheet.metrics[].value_text", "fact_sheet.metrics[].ci_low", "fact_sheet.score", "client_inputs"):
        assert not any(d == path or d.startswith(f"{path}.") for d in diff), path


def _issued_edition3(tag: str) -> dict:
    return json.loads((TESTS / "data" / "edition3_issued" / "requests" / f"{tag}.request.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_differs_from_its_edition3_request_only_at_declared_places(name):
    """세 예시 모두(다중·멀티 포함) — 같은 입력의 판 3 요청과 다른 자리는 명세가 판 4 키·판 4 표기·판 4 뜻으로 적은 칸뿐."""
    ex = _load(name)
    declared = set(ED4_ONLY) | {k for k in SPEC["edition4_notation"] if not k.startswith("$")} | set(SPEC["edition4_meaning"])
    diff = _diff_paths(_issued_edition3(EXAMPLES[name]), ex)
    assert "report_edition" in diff and "fact_sheet.metrics[].ci_low_text" in diff
    # 선언된 칸이거나, 선언된 칸 아래(고정 문장 목록의 글 — 길이가 같아 칸 하나로 나온다), 또는 선언된 칸을 품은 목록(길이가 달라짐).
    def declared_here(p: str) -> bool:
        return any(p == d or p.startswith(f"{d}.") or p.startswith(f"{d}[]") or d.startswith(f"{p}[]") for d in declared)

    stray = sorted(p for p in diff if not declared_here(p))
    assert stray == []
    for path in ("fact_sheet.metrics[].value_text", "fact_sheet.metrics[].ci_low", "fact_sheet.score", "client_inputs"):
        assert not any(d == path or d.startswith(f"{path}.") for d in diff), path


def _lost_keys(a, b, path: str = "") -> list[str]:
    """a 의 키·값이 b 에 그대로 있는지(b 쪽에만 있는 기본값 칸은 보지 않는다 — 스키마가 채운 None). 1 == 1.0 은 같다."""
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return [path]
        return [x for k, v in a.items() for x in ([f"{path}.{k}"] if k not in b else _lost_keys(v, b[k], f"{path}.{k}"))]
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return [path]
        return [x for i, (v, w) in enumerate(zip(a, b)) for x in _lost_keys(v, w, f"{path}[{i}]")]
    return [] if a == b else [f"{path} ({a!r} != {b!r})"]


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_every_example_is_an_edition4_request_the_schema_keeps_whole(name):
    """판 신호 4, 18 없음, 지표 줄마다 경계 표기 키, 판정 여유 불변식, 스키마를 지나도 버려지는 키·잘리는 칸 없음."""
    from app.narrative.jobs import parse_request
    from app.narrative.schemas import NarrativeRequest

    ex = _load(name)
    assert ex["report_edition"] == 4
    assert "18" not in [s["id"] for s in ex["fact_sheet"]["facts"]["fixed_sentences"]]
    metrics = ex["fact_sheet"]["metrics"]
    assert metrics and all("ci_low_text" in m and "ci_high_text" in m for m in metrics)
    with_ci = [m for m in metrics if m["ci_contains_target"] is not None]
    assert with_ci
    for m in with_ci:
        assert m["near_target"] is m["ci_contains_target"], m["metric_id"]
    assert [p["kind"] for p in ex["fact_sheet"]["problems"]][-1] == "same_source"
    assert _lost_keys(ex, NarrativeRequest.model_validate(ex).model_dump(mode="json")) == []
    req, trimmed = parse_request(json.loads(json.dumps(ex)))
    assert (req.report_edition, trimmed) == (4, [])


def test_spec_bc_strong_values_match_example():
    by_id = {m["metric_id"]: m for m in EX["fact_sheet"]["metrics"]}
    checked = 0
    for entry in SPEC["edition4_only_keys"]:
        path, want = entry["path"], entry["bc_strong"]
        if path.startswith("fact_sheet.metrics[]."):
            key = path.rsplit(".", 1)[1]
            for mid, m in by_id.items():
                assert m[key] == want.get(mid, want["그 밖"]), (path, mid)
                checked += 1
        elif path.startswith("fact_sheet.facts.distribution_shift[]."):
            key = path.rsplit(".", 1)[1]
            rows = EX["fact_sheet"]["facts"]["distribution_shift"]
            assert {r["label"]: r[key] for r in rows} == want
            checked += 1
        else:
            [(node, key)] = _containers(EX, path)
            assert node[key] == want, path
            checked += 1
    assert checked >= 20


# ── 대조 표: 기준 구현 ─────────────────────────────────────────────────────────


def _fixed(value, digits: int) -> str:
    """이진 값 그대로의 절반에서 0에서 먼 쪽(JS toFixed 와 같다)."""
    return format(Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), "f")


def _unneg(text: str) -> str:
    return text[1:] if text.startswith("-") and set(text[1:]) <= set("0.") else text


def _diff(a: float, b: float) -> float:
    return round(float(a) - float(b), 10)


def _meets(value: float, threshold: float, higher_is_better: bool) -> bool:
    d = _diff(value, threshold)
    return d >= 0 if higher_is_better else d <= 0


def ref_plain(value) -> str:
    return _unneg(_fixed(value, 3))


def ref_judged(value: float, threshold: float, higher_is_better: bool, meets: bool) -> str:
    for digits in range(3, 7):
        text = _fixed(value, digits)
        if _meets(float(text), threshold, higher_is_better) == meets:
            return _unneg(text)
    scale = 10**6
    toward_pass = higher_is_better == meets
    cut = (math.ceil(value * scale) if toward_pass else math.floor(value * scale)) / scale
    return _unneg(_fixed(cut, 6))


def ref_signed_judged(value: float, threshold: float, meets: bool) -> str:
    magnitude = ref_judged(abs(value), threshold, True, meets)
    return f"-{magnitude}" if value < 0 and set(magnitude) - set("0.") else magnitude


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def ref_ci_bound(raw: float, target: float) -> str:
    want = _sign(_diff(raw, target))
    for digits in range(3, 7):
        text = _fixed(raw, digits)
        if _sign(_diff(float(text), target)) == want:
            return _unneg(text)
    scale = 10**6
    cut = (math.floor(raw * scale) if want < 0 else math.ceil(raw * scale)) / scale
    return _unneg(_fixed(cut, 6))


def _moved(text: str) -> tuple[str, str]:
    sign = "-" if text.startswith("-") else ""
    whole, _, frac = text.lstrip("-").partition(".")
    digits = whole + frac.ljust(2, "0")
    point = len(whole) + 2
    head = digits[:point].lstrip("0") or "0"
    tail = digits[point:] or "0"
    return sign, f"{head}.{tail}"


def ref_percent(text: str) -> str:
    sign, body = _moved(text)
    return f"{sign if set(body) - set('0.') else ''}{body}%"


def ref_percent_point(text: str) -> str:
    sign, body = _moved(text)
    if not set(body) - set("0."):
        return f"{body}%p"
    return f"{'−' if sign else '+'}{body}%p"


def ref_signed_count(n: int) -> str:
    return f"+{n}" if n > 0 else f"−{-n}" if n < 0 else "0"


def _shortest(value) -> Decimal:
    return Decimal(repr(float(value)))


def ref_given(value) -> str:
    d = _shortest(value)
    return str(int(d)) if d == int(d) else format(d, "f")


def ref_target(value) -> str:
    d = _shortest(value)
    decimals = 0 if d == int(d) else -d.as_tuple().exponent
    if decimals <= 3:
        return ref_plain(value)
    return format(d, "f") if decimals <= 6 else _fixed(value, 6)


def ref_latency(value) -> str:
    return _fixed(value, 2)


def ref_distinct(value, others) -> str:
    rivals = [o for o in others if _fixed(o, 6) != _fixed(value, 6)]
    for digits in range(3, 7):
        if all(_fixed(o, digits) != _fixed(value, digits) for o in rivals):
            return _unneg(_fixed(value, digits))
    raise AssertionError("여섯째 자리로 다르면 여섯째 자리에서 끝난다")


def ref_number(case: dict) -> str:
    rule = case["rule"]
    if rule == "plain":
        return ref_plain(case["value"])
    if rule == "judged":
        return ref_judged(case["value"], case["threshold"], case["higher_is_better"], case["meets"])
    if rule == "signed_judged":
        return ref_signed_judged(case["value"], case["threshold"], case["meets"])
    if rule == "ci_bound":
        return ref_ci_bound(case["raw"], case["target"])
    if rule == "percent":
        return ref_percent(case["text"])
    if rule == "percent_point":
        return ref_percent_point(case["text"])
    if rule == "signed_count":
        return ref_signed_count(case["value"])
    if rule == "given":
        return ref_given(case["value"])
    if rule == "target":
        return ref_target(case["value"])
    if rule == "latency":
        return ref_latency(case["value"])
    if rule == "distinct":
        return ref_distinct(case["value"], case["others"])
    raise AssertionError(f"모르는 규칙 {rule}")


RULES = (
    "plain", "judged", "signed_judged", "ci_bound", "percent", "percent_point", "signed_count", "given", "target", "latency",
    "distinct",
)


def test_table_covers_every_rule_with_several_cases():
    counts = {rule: sum(1 for c in TABLE["number_cases"] if c["rule"] == rule) for rule in RULES}
    assert {c["rule"] for c in TABLE["number_cases"]} == set(RULES)
    assert min(counts.values()) >= 4, counts
    assert set(TABLE["rules"]) >= set(RULES)
    assert TABLE["functions"] == SPEC["function_names"]


@pytest.mark.parametrize("case", TABLE["number_cases"], ids=lambda c: f"{c['rule']}:{c['expected']}")
def test_number_table_follows_the_rule(case):
    assert ref_number(case) == case["expected"]


def test_reference_implementation_catches_wrong_table_rows():
    """기준 구현이 이빨이 있는지 — 표의 기대값을 하나씩 비틀면 모두 어긋나야 한다."""
    wrong = 0
    for case in TABLE["number_cases"]:
        bent = {**case, "expected": case["expected"] + "0"}
        wrong += ref_number(bent) != bent["expected"]
    assert wrong == len(TABLE["number_cases"])
    # 파이썬 round() 로 찍으면 표와 어긋나는 줄이 있다(반올림 규칙을 표가 정말 고정한다).
    assert f"{round(0.0625, 3):.3f}" == "0.062"
    assert any(c["rule"] == "plain" and c["value"] == 0.0625 and c["expected"] == "0.063" for c in TABLE["number_cases"])


def test_text_rules_give_number_shaped_texts():
    """사실 시트 *_text 에 싣는 규칙의 결과는 숫자 꼴이다(판 3 NumText 와 같은 꼴)."""
    import re

    num = re.compile(r"^-?\d+(\.\d+)?$")
    rows = [c for c in TABLE["number_cases"] if c["rule"] in ("plain", "judged", "signed_judged", "ci_bound", "distinct", "target", "given", "latency")]
    assert rows
    assert [c["expected"] for c in rows if not num.match(c["expected"])] == []
    # 그리는 규칙(percent·percent_point·signed_count)의 결과는 숫자 꼴이 아니다 — *_text 칸에 싣지 않는다.
    drawn = [c for c in TABLE["number_cases"] if c["rule"] in ("percent", "percent_point", "signed_count") and c["expected"] != "0"]
    assert drawn and all(not num.match(c["expected"]) for c in drawn)


@pytest.mark.parametrize("case", TABLE["josa_cases"], ids=lambda c: f"{c['word']}{c['expected']}")
def test_josa_table_matches_backend_josa(case):
    assert josa(case["word"], case["pair"]) == case["expected"]


def test_josa_table_has_numbers_and_names_for_every_pair():
    pairs = {(c["kind"], c["pair"]) for c in TABLE["josa_cases"]}
    for kind in ("number", "name"):
        for pair in ("이/가", "을/를", "은/는", "과/와", "으로/로"):
            assert (kind, pair) in pairs, (kind, pair)


@pytest.mark.parametrize("chain", TABLE["chains"], ids=lambda c: c["expected"])
def test_chains_number_then_josa(chain):
    text = ref_number({**chain["number"]})
    assert text + josa(text, chain["pair"]) == chain["expected"]
    if chain["edition3_text"]:
        old = chain["edition3_text"]
        number = old.rstrip("으로를을가이은는과와")
        assert number != text and number + josa(number, chain["pair"]) == old


def test_example_numbers_are_all_in_the_table():
    produced = set()
    for case in TABLE["number_cases"]:
        if case.get("trap"):
            continue
        produced.add(case["expected"])
        produced.add(case["expected"].lstrip("-"))
    produced |= {e["expected"] for e in TABLE["engine_texts"]}
    assert TABLE["example_numbers"]
    assert sorted(set(TABLE["example_numbers"]) - produced) == []


# ── 임시 판 4 예시의 글자 = 대조 표 규칙 + S1 원값 ──────────────────────────────


EVAL = json.loads((TESTS / "golden_ed4" / "evaluate_bc_strong.json").read_text(encoding="utf-8"))["results"]["success_metrics"]


def test_example_metric_texts_follow_rules():
    metrics = EX["fact_sheet"]["metrics"]
    judged = [m for m in metrics if m["threshold"] is not None]
    assert len(judged) == 5
    for m in metrics:
        mid = m["metric_id"]
        if m["threshold"] is None:
            assert (m["ci_low_text"], m["ci_high_text"], m["near_target"]) == (None, None, False), mid
            continue
        ci = EVAL["confidence_intervals"][mid]
        assert m["ci_low_text"] == ref_ci_bound(ci["low_raw"], m["threshold"]), mid
        assert m["ci_high_text"] == ref_ci_bound(ci["high_raw"], m["threshold"]), mid
        contains = _diff(ci["low_raw"], m["threshold"]) <= 0 <= _diff(ci["high_raw"], m["threshold"])
        assert m["ci_contains_target"] is contains, mid
        assert m["near_target"] is m["ci_contains_target"], mid  # 판 4 뜻: 구간 기준 판정 여유
        assert m["margin_text"] == ref_judged(m["margin"], 0, True, m["status"] == "pass"), mid
        assert m["value_text"] == ref_judged(m["value"], m["threshold"], m["higher_is_better"], m["status"] == "pass"), mid
    # 판 3 와 달라진 판정 여유: M1 은 작음(구간이 목표 포함), M9 는 충분(3% 안이지만 구간이 떨어짐).
    by_id = {m["metric_id"]: m for m in metrics}
    assert (by_id["M1"]["near_target"], by_id["M9"]["near_target"]) == (True, False)


def test_example_threshold_gain_texts_follow_rules():
    g = EX["fact_sheet"]["f1_threshold_gain"]
    raw = EVAL["f1_threshold_gain"]
    by_id = {m["metric_id"]: m for m in EX["fact_sheet"]["metrics"]}

    def text(value, mid):
        m = by_id.get(mid)
        if m is None or m["threshold"] is None:
            return ref_plain(value)
        return ref_judged(value, m["threshold"], m["higher_is_better"], _meets(value, m["threshold"], m["higher_is_better"]))

    assert g["max_f1_text"] == text(raw["max_f1_raw"], "M4")
    assert g["current_f1_text"] == text(raw["current_f1_raw"], "M4") == by_id["M4"]["value_text"]
    assert g["best_recall_text"] == text(raw["best_recall_raw"], "M3")
    assert g["current_recall_text"] == text(raw["current_recall_raw"], "M3") == by_id["M3"]["value_text"]
    assert g["best_precision_text"] == text(raw["best_precision_raw"], "M2")
    assert g["current_precision_text"] == text(raw["current_precision_raw"], "M2")
    assert g["gain_text"] == ref_judged(raw["gain_raw"], 0.1, True, raw["notable"])
    assert g["best_threshold_text"] == raw["best_threshold_text"]
    target = by_id["M3"]["threshold"]
    assert g["risk_side"] == {
        "side": "fn",
        "metric_id": "M3",
        "display_name": "Recall",
        "meets_target_now": _meets(raw["current_recall_raw"], target, True),
        "meets_target_at_best": _meets(raw["best_recall_raw"], target, True),
    }


def test_example_imbalance_and_facts_texts_follow_rules():
    d = EX["fact_sheet"]["imbalance_effect"]["display"]
    raw = EVAL["imbalance_effect"]
    assert d["imbalance_ratio"] == ref_judged(raw["imbalance_ratio_raw"], 1.5, True, raw["level"] != "mild")
    for key in ("accuracy_minus_macro_f1", "majority_minus_minority_f1"):
        v = raw[f"{key}_raw"]
        assert d[key] == ref_judged(v, 0.1, True, _meets(v, 0.1, True)), key
    for key in ("accuracy", "macro_f1", "majority_f1", "minority_f1"):
        assert d[key] == ref_plain(raw[f"{key}_raw"]), key
    f = EX["fact_sheet"]["facts"]
    w = f["weakest_class"]
    assert w["share_text"] == ref_judged(w["share"], 0.05, True, w["share"] >= 0.05)
    assert w["gap_text"] == ref_judged(w["gap_from_macro"], 0.1, True, w["far_below_macro"])
    assert f["declared_risk_errors"]["share_text"] == ref_plain(f["declared_risk_errors"]["share"])
    pr = f["precision_recall_gap"]
    assert pr["difference_text"] == ref_signed_judged(pr["difference"], 0.1, pr["flagged"])
    rows = f["distribution_shift"]
    assert len(rows) == 2
    for row in rows:
        assert row["difference_text"] == ref_signed_judged(row["difference"], 0.1, row["flagged"])
        gap = row["evaluation_ratio"] - row["training_ratio"]
        assert row["evaluation_minus_training_text"] == ref_signed_judged(gap, 0.1, row["flagged"])
    # 4.5 차이 열(평가 − 학습)로 그리면 예시본 글자.
    assert [ref_percent_point(r["evaluation_minus_training_text"]) for r in rows] == ["−0.2%p", "+0.2%p"]
    ids = [s["id"] for s in f["fixed_sentences"]]
    assert ids and "18" not in ids
    nine = next(s for s in f["fixed_sentences"] if s["id"] == "9")
    assert nine == {"id": "9", "section": "7.1", "text": "본 평가의 예측값은 제출된 확률값에 결정 임계값 0.5를 적용하여 산출하였습니다."}


def test_example_same_source_problem_follows_spec():
    problems = EX["fact_sheet"]["problems"]
    shape = SPEC["problems"]["same_source"]["shape"]
    same = [p for p in problems if p["kind"] == "same_source"]
    assert same == [{**shape, "id": "P2"}]
    assert problems[-1]["kind"] == "same_source"  # LOW 맨 끝
    assert [p["id"] for p in problems] == [f"P{i}" for i in range(1, len(problems) + 1)]
    assert problems[0]["title"] == "핵심 지표 근소 합격 (Recall)"
    # 같은 출처 판정 = 고정 문구 11a 와 같은 판정.
    assert any(s["id"] == "11a" for s in EX["fact_sheet"]["facts"]["fixed_sentences"])


# ── 칸 → 자리 ────────────────────────────────────────────────────────────────────


def test_slots_cover_backend_sections():
    slots = [s["slot"] for s in SLOTS["slots"]]
    assert slots and set(slots) == set(SECTION_IDS)
    assert sorted(SLOTS["order_edition4"]) == sorted(slots) == sorted(SLOTS["order_edition3"])
    assert list(SECTION_IDS) == SLOTS["order_edition3"]
    assert [s["slot"] for s in SLOTS["slots"]] == SLOTS["order_edition4"]
    places = [s["edition4_place"] for s in SLOTS["slots"]]
    assert places == ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "6"]
    assert [(r["slot"], r["edition4_place"]) for r in SLOTS["recommend_sections"]] == [("9.1", "6.1"), ("9.2", "6.2")]
    assert set(places) | {"6.1", "6.2"} <= set(SLOTS["edition4_section_numbers"])


def test_record_panel_lines_are_the_approved_b26_text():
    lines = SLOTS["record_panel"]["lines"]
    assert " / ".join(line["text"] for line in lines) == (
        "5.1 판정과 불확실성(8.2 칸) / 5.2 오분류(7.1 칸) / 5.3 데이터 분포(7.2 칸) / 5.4 참조 범위(8.1 칸) / "
        "5.5 운영 위험(8.3 칸) / 5.6 추가 관찰(7.3 칸) / 6 개선 권고(9 칸)"
    )
    for line, slot in zip(lines, SLOTS["slots"]):
        assert line["slot"] == slot["slot"]
        assert line["text"] == f"{slot['edition4_place']} {slot['edition4_title']}({slot['slot']} 칸)"
    assert SLOTS["record_panel"]["header"] + SLOTS["record_panel"]["header_note"] == (
        "작성 기록 · 5·6절 칸마다 쓴 방식(화면에만 보이며 인쇄본에는 없습니다)"
    )


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_example_weakest_f1_text_follows_d2_then_d3(name):
    """facts.weakest_class.f1_text(판 4 전용) — 판정 대상 지표와 같은 원값이면 그 value_text(D2), 아니면 distinct(다른 클래스 F1
    — D3). 기준 구현으로 본다(프론트가 다시 떠도 이 규칙이어야 한다)."""
    fs = _load(name)["fact_sheet"]
    w = fs["facts"]["weakest_class"]
    same = [m["value_text"] for m in fs["metrics"]
            if m.get("value_text") and m["threshold"] is not None and abs(m["value"] - w["f1"]) < 1e-9]
    want = same[0] if same else ref_distinct(w["f1"], [p["f1"] for p in fs["per_class"] if p["label"] != w["label"]])
    assert w["f1_text"] == want
    assert {"fact_sheet_binary_ed4.json": "0.977", "fact_sheet_multiclass_ed4.json": "0.9720",
            "fact_sheet_multilabel_ed4.json": "0.048"}[name] == want
    if name == "fact_sheet_binary_ed4.json":
        assert same == ["0.977"]  # 이진 — 양성 클래스 F1 = F1 Score(M4)
