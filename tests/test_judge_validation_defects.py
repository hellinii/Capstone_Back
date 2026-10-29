"""결함본·sham 파일 꼴과 검사기, 사람 채점지·열쇠(계획 17-5, 설계 7.2, Q12) — 심판 라운드 J8.

바탕은 합성 문장(봉인 서술은 쓰지 않는다 — 검사기의 봉인 열람은 V1 뒤 verify 전용, 그 거부도 여기서 본다).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.judge import criteria
from app.judge.validation import sheets as SH
from app.judge.validation import variants as V

ROOT = Path(__file__).resolve().parents[1]


def _jv():
    if "judge_validate" in sys.modules:
        return sys.modules["judge_validate"]
    spec = importlib.util.spec_from_file_location("judge_validate", ROOT / "scripts" / "judge_validate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


def synthetic_result() -> dict:
    """합성 서술 결과 — 작업 결과 result 모양(칸 글은 합성 문장)."""
    return {
        "interpretation": {"confusion_analysis": "합성 문장 가. 합성 문장 나는 0.91 이다. 합성 문장 다.",
                           "distribution_analysis": "분포 합성 문장.", "additional_observations": ""},
        "conclusion": {"verdict": "PASS", "benchmark": "참조 합성 문장.", "narrative": "총평 합성 하나. 총평 합성 둘.",
                       "risks": "위험 합성 문장."},
        "recommendation_narrative": {"data_quality": "데이터 권고 합성.", "model_ops": "운영 권고 합성."},
        "recommendations": [{"problem_id": "P1", "priority": "HIGH", "category": "점검", "action": "합성 조치.",
                             "expected_impact": "합성 효과."}],
        "section_sources": {k: "llm" for k in ("7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "9")},
    }


def _meta(**kw) -> dict:
    meta = {"id": "D01", "base": "syn", "axis": "faithfulness", "kind": "숫자 바꾸기", "planted_by": "민서",
            "slot": "interpretation.confusion_analysis", "intent": "0.91 을 0.81 로", "before": "합성 문장 나는 0.91 이다.",
            "after": "합성 문장 나는 0.81 이다."}
    meta.update(kw)
    return meta


def _new_text(meta=None) -> str:
    return V.replace_sentence(synthetic_result()["interpretation"]["confusion_analysis"], 1, "합성 문장 나는 0.81 이다.")


def _check(meta=None, slots=None, diff="auto", base=None, name=None):
    base = base or synthetic_result()
    meta = meta or _meta()
    slots = {meta["slot"]: _new_text()} if slots is None else slots
    if diff == "auto":
        text = next(iter(slots.values())) if slots and isinstance(next(iter(slots.values())), str) else ""
        diff = V.make_diff(V.read_slot(base, meta["slot"]), text, meta["slot"]) if meta.get("slot") in V.slot_paths(base) else ""
    return V.check_variant(name or meta.get("id", "D01"), meta, slots, diff, base)


# ── 칸·적용 ──


def test_slot_paths_and_apply_leave_base_untouched():
    base = synthetic_result()
    paths = V.slot_paths(base)
    assert "conclusion.narrative" in paths and "recommendations.0.action" in paths and "conclusion.verdict" not in paths
    changed = V.apply_slots(base, {"recommendations.0.action": "바꾼 조치."})
    assert changed["recommendations"][0]["action"] == "바꾼 조치." and base["recommendations"][0]["action"] == "합성 조치."
    with pytest.raises(KeyError):
        V.apply_slots(base, {"conclusion.verdict": "FAIL"})  # 서술 칸이 아니다
    assert SH.slot_of_path("conclusion.narrative") == "8.2" and SH.slot_of_path("recommendations.0.action") == "9"


# ── 검사기 — 통과 셋(바꿈·지움·끼움)과 조건마다 ──


def test_valid_replace_delete_insert_pass():
    assert _check() == []
    gone = V.replace_sentence(synthetic_result()["interpretation"]["confusion_analysis"], 1, None)
    assert _check(_meta(axis="coverage", planted_by="인용", after="", item_id="분석-7"),
                  {"interpretation.confusion_analysis": gone}) == []
    meta = _meta(id="S01", axis="sham", planted_by="인용", slot="conclusion.narrative", before="", after="총평 합성 셋.")
    assert _check(meta, {"conclusion.narrative": "총평 합성 하나. 총평 합성 둘. 총평 합성 셋."}) == []


@pytest.mark.parametrize("field", V.META_FIELDS)
def test_missing_meta_field(field):
    meta = _meta()
    del meta[field]
    assert _check(meta, slots={"interpretation.confusion_analysis": _new_text()}, diff="") == [f"meta.json 에 '{field}' 칸이 없다"]


def test_empty_meta_field():
    assert _check(_meta(intent=" ")) == ["meta.json 의 'intent' 가 비었다"]


@pytest.mark.parametrize("meta, name, expected", [
    (_meta(id="D02"), "D01", "폴더 이름과 meta.id"),
    (_meta(id="X1"), "X1", "폴더 이름과 meta.id"),
    (_meta(axis="truth"), None, "네 축 중 하나가 아니다"),
    (_meta(axis="coverage", planted_by="민서", item_id="분석-7"), None, "coverage 결함은 인용 이 심는다"),
    (_meta(axis="coverage", planted_by="인용"), None, "meta.item_id"),
    (_meta(planted_by="인용"), None, "faithfulness 결함은 민서 이 심는다"),
    (_meta(round=2), None, "faithfulness 결함은 인용 이 심는다(설계 7.2 역할, 다시 잼)"),
    (_meta(id="S01", axis="faithfulness"), None, "sham 의 축은 'sham'"),
    (_meta(id="S01", axis="sham", planted_by="누군가"), None, "민서·인용 중 하나"),
])
def test_meta_rules(meta, name, expected):
    problems = _check(meta, name=name)
    assert len(problems) == 1 and expected in problems[0], problems


def test_retry_round_swaps_roles():
    # 다시 잴 때(Q12 ④): 인용 님이 모든 결함을 심고 민서가 채점한다.
    assert _check(_meta(round=2, planted_by="인용")) == []
    assert (V.scorer_for("faithfulness", 2), V.scorer_for("faithfulness"), V.scorer_for("coverage")) == ("민서", "인용", "민서")


def test_more_than_one_slot_or_wrong_key():
    two = {"interpretation.confusion_analysis": _new_text(), "conclusion.risks": "위험 합성 바꿈."}
    assert _check(slots=two, diff="") == ["slots.json 은 meta.slot 칸 하나만 담는다(담긴 칸 2개)"]
    assert _check(slots={"conclusion.risks": "x."}, diff="") == ["slots.json 은 meta.slot 칸 하나만 담는다(담긴 칸 1개)"]


def test_slot_not_in_base():
    meta = _meta(slot="conclusion.verdict")
    assert _check(meta, slots={"conclusion.verdict": "FAIL"}, diff="") == ["바탕 성적서에 없는 서술 칸이다: conclusion.verdict"]


def test_two_sentences_changed_or_nothing_changed():
    text = synthetic_result()["interpretation"]["confusion_analysis"]
    twice = V.replace_sentence(V.replace_sentence(text, 0, "바꾼 가."), 2, "바꾼 다.")
    assert "문장 하나만 바뀐 것이 아니다" in _check(slots={"interpretation.confusion_analysis": twice})[0]
    assert "문장 하나만 바뀐 것이 아니다" in _check(slots={"interpretation.confusion_analysis": text})[0]


def test_two_neighbouring_sentences_replaced_is_not_one():
    # 붙은 두 문장을 한 번에 바꾸면(difflib 바꿈 한 덩어리, 2→2) 문장 하나가 아니다.
    text = synthetic_result()["interpretation"]["confusion_analysis"]
    both = V.replace_sentence(V.replace_sentence(text, 0, "바꾼 가."), 1, "바꾼 나.")
    assert V.sentence_change(text, both) is None
    assert "문장 하나만 바뀐 것이 아니다" in _check(slots={"interpretation.confusion_analysis": both})[0]


def test_before_after_must_match_letter_for_letter():
    assert _check(_meta(before="합성 문장 나는 0.9 이다.")) == [
        "meta.before 가 바탕 칸에서 바뀐(지운) 문장과 글자 그대로 같지 않다"]
    assert _check(_meta(after="합성 문장 나는 0.71 이다.")) == ["meta.after 가 새로 들어간 문장과 글자 그대로 같지 않다"]


def test_diff_missing_or_stale():
    assert _check(diff=None) == ["diff.txt 가 없다"]
    assert _check(diff="손으로 쓴 차이\n") == ["diff.txt 가 다시 만든 차이와 다르다(도구로 다시 쓴다)"]


def test_problem_messages_do_not_quote_narrative():
    # 화면에 찍히는 문제 문장에 서술 글이 옮겨지지 않는다(봉인 내용 보호).
    problems = _check(_meta(before="엉뚱", after="엉뚱"), slots={"interpretation.confusion_analysis": "완전히 다른 글. 둘째."})
    assert problems and not any("합성" in p or "완전히" in p for p in problems)


def test_write_variant_round_trip(tmp_path):
    base = synthetic_result()
    folder = V.write_variant(tmp_path, _meta(), base, _new_text())
    meta, slots = V.load_variant(folder)
    assert V.check_variant("D01", meta, slots, (folder / "diff.txt").read_text(encoding="utf-8"), base) == []
    assert [p.name for p in V.variant_folders(tmp_path, "D")] == ["D01"] and V.variant_folders(tmp_path, "S") == []


# ── 봉인 열람 거부(verify 전용 · V1 뒤 · 사람 허락) ──


def test_check_defects_verify_refuses_before_v1_or_without_permission(monkeypatch):
    jv = _jv()
    with pytest.raises(jv.RunnerError) as err:
        jv.cmd_check_defects("verify", allow_sealed=False, echo=lambda _: None)
    assert "--allow-sealed 가 없다" in str(err.value) and "기준 버전이 고정되지 않았다" in str(err.value)
    with pytest.raises(jv.RunnerError, match="기준 버전이 고정되지 않았다"):
        jv.cmd_check_defects("verify", allow_sealed=True, echo=lambda _: None)
    monkeypatch.setattr(criteria, "version", lambda: "1.0")
    monkeypatch.setattr(criteria, "version_problem", lambda *a, **k: None)
    with pytest.raises(jv.RunnerError, match="--allow-sealed 가 없다"):
        jv.cmd_check_defects("verify", allow_sealed=False, echo=lambda _: None)


def test_cli_verify_refuses_dev_paths(capsys):
    jv = _jv()
    assert jv.main(["check-defects", "--mode", "verify", "--defects", "x"]) == 2
    assert "dev 전용" in capsys.readouterr().err


def test_check_defects_dev_on_synthetic_variants(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(tmp_path / "검증"))
    jv = _jv()
    import asyncio

    base = asyncio.run(jv.dev_bases(jv.DEV_INPUTS))["dev_binary"]
    slot = "conclusion.narrative"
    text = V.read_slot(base.result, slot)
    first = V.sentences(text)[0]
    meta = {"id": "D01", "base": "dev_binary", "axis": "readability", "kind": "영어 코드값 노출", "planted_by": "인용",
            "slot": slot, "intent": "코드값 worse 노출", "before": first, "after": first[:-1] + " (worse)."}
    V.write_variant(tmp_path / "결함", meta, base.result, V.replace_sentence(text, 0, meta["after"]))
    bad = dict(meta, id="D02", planted_by="민서")
    V.write_variant(tmp_path / "결함", bad, base.result, V.replace_sentence(text, 0, meta["after"]))
    lines: list[str] = []
    problems = jv.cmd_check_defects("dev", defects=tmp_path / "결함", echo=lines.append)
    assert lines[0] == "D01 통과" and lines[1] == "D02 문제 1개"
    assert problems == ["D02: readability 결함은 인용 이 심는다(설계 7.2 역할) — 적힌 것: 민서"]
    (tmp_path / "결함" / "D01" / "diff.txt").write_text("x", encoding="utf-8")
    jv.cmd_make_diff("dev", "D01", defects=tmp_path / "결함", echo=lines.append)
    assert jv.cmd_check_defects("dev", defects=tmp_path / "결함", echo=lines.append) == problems


# ── 채점지·열쇠 ──


TYPES = criteria.load_json("contradictions.json")["types"]


def _run(run_id, variant="base", axis="", slot="", problem=False):
    """합성 회차 결과 — 축마다 문항이 나오게."""
    rec = {
        "faithfulness": {"verdicts": [{"claim_id": f"C{i}", "kind": "number", "verdict": "violation" if (problem and i == 0) else "match",
                                       "slot": "7.1", "piece_id": "slot:7.1", "quote": f"합성 {i}"} for i in range(6)]},
        "coverage": {"items": [{"id": f"분석-{i}", "described": not (problem and i == 1), "piece_id": "", "slot": "", "place": "",
                                "quote": ""} for i in range(1, 7)]},
        "consistency": {"contradictions": [{"type": "analysis_vs_finding", "pieces": [], "slots": [], "sources": [], "quotes": []}]
                        if problem else []},
        "readability": {"findings": [{"rule": "r", "piece_id": "slot:8.2", "slot": "8.2", "text": "x"}] if problem else [],
                        "llm_flags": []},
        "pieces": [{"id": f"slot:{s}", "slot": s, "place": s, "source": "ai"} for s in ("7.1", "7.2", "8.2", "9")],
    }
    meta = {"axis": axis, "slot": slot} if variant != "base" else {}
    return {"run_id": run_id, "key": {"batch": "x", "report": "bc", "variant": variant, "repeat": 1},
            "variant_meta": meta, "scorecard": {"x": 1}, "record": rec}


def _items():
    runs = [_run("R-1"), _run("R-2", problem=True), _run("R-3", "D01", "faithfulness", "interpretation.confusion_analysis", True),
            _run("R-4", "D04", "coverage", "conclusion.narrative", True), _run("R-5", "S01", "sham", "conclusion.risks")]
    return [i for r in runs for i in SH.items_from_run(r, TYPES)]


def test_items_follow_q12_units():
    items = SH.items_from_run(_run("R-1"), TYPES)
    by = {a: [i for i in items if i["axis"] == a] for a in V.AXES}
    assert len(by["faithfulness"]) == 6 and len(by["coverage"]) == 6
    assert [i["unit"] for i in by["consistency"]] == [t["id"] for t in TYPES]  # 성적서 하나 × 모순 유형
    assert [i["unit"] for i in by["readability"]] == ["7.1", "7.2", "8.2", "9"]  # 칸 하나
    assert SH.items_from_run({**_run("R-9"), "scorecard": None}, TYPES) == []  # 채점 실패 회차는 문항 없음


def test_sample_is_seeded_stratified_and_keeps_planted_items():
    items = _items()
    a, b = SH.sample(items, 20, seed=3), SH.sample(items, 20, seed=3)
    assert a == b and SH.sample(items, 20, seed=4) != a
    assert {ax: sum(i["axis"] == ax for i in a) for ax in V.AXES} == {ax: 5 for ax in V.AXES}
    assert any(i["kind"] == "defect" and i["axis"] == "faithfulness" for i in a)
    assert any(i["kind"] == "defect" and i["axis"] == "coverage" for i in a)
    for ax in V.AXES:  # 축마다 '문제' 층이 있으면 뽑힌다
        assert any(i["problem"] for i in a if i["axis"] == ax)


@pytest.mark.parametrize("seed", range(5))
def test_smallest_sample_takes_the_planted_problem_item_first(seed):
    # 축마다 한 문항이면 심은 자리에서 시스템이 문제라고 한 문항이 뽑힌다(탐지한 결함 사례를 표본에 꼭 넣는다).
    got = {i["axis"]: i for i in SH.sample(_items(), 4, seed=seed)}
    assert (got["faithfulness"]["run_id"], got["faithfulness"]["unit"]) == ("R-3", "C0")
    assert (got["coverage"]["run_id"], got["coverage"]["unit"], got["coverage"]["problem"]) == ("R-4", "분석-1", True)


def test_assign_follows_roles_and_overlap_avoids_defects():
    got = SH.assign(SH.sample(_items(), 40, seed=5), overlap=10, seed=5)
    for i in got:
        if i["kind"] == "defect":
            assert i["scorers"] == [V.scorer_for(i["variant_axis"])]  # F·Cons → 인용, Cov·R → 민서
    assert {i["scorers"][0] for i in got if i["kind"] == "defect" and i["variant_axis"] == "coverage"} == {"민서"}
    both = [i for i in got if len(i["scorers"]) == 2]
    assert len(both) == 10 and all(i["kind"] != "defect" for i in both)
    # V1 결정 7: 겹침 10문항은 모두 합친 수 — 축마다 고르게 3·3·2·2
    assert [sum(i["axis"] == a for i in both) for a in V.AXES] == [3, 3, 2, 2]


def test_overlap_quota_spreads_and_moves_leftover():
    assert SH.overlap_quota(10, {a: 10 for a in V.AXES}) == dict(zip(V.AXES, (3, 3, 2, 2)))
    short = dict(zip(V.AXES, (1, 10, 10, 10)))  # 첫 축 후보가 1개 — 남는 몫은 다른 축으로
    q = SH.overlap_quota(10, short)
    assert q[V.AXES[0]] == 1 and sum(q.values()) == 10
    assert sum(SH.overlap_quota(10, dict(zip(V.AXES, (1, 1, 1, 1)))).values()) == 4  # 후보가 모자라면 있는 만큼
    assert SH.overlap_quota(0, {a: 5 for a in V.AXES}) == {a: 0 for a in V.AXES}


def test_sheets_are_blind_and_key_is_separate():
    assigned = SH.assign(SH.sample(_items(), 40, seed=5), overlap=8, seed=5)
    sheets, key, docs = SH.build(assigned, seed=5)
    flat = json.dumps(sheets, ensure_ascii=False)
    for secret in ("D01", "D04", "S01", "R-1", "R-3", "violation", "match", "결함", "sham", "base"):
        assert secret not in flat
    assert all(len(r) == len(SH.SHEET_COLUMNS) and r[6] == "" for rows in sheets.values() for r in rows)
    assert len(key) == len(assigned) and {k[0] for k in key} == {r[0] for rows in sheets.values() for r in rows}
    assert set(docs) == {r[1] for rows in sheets.values() for r in rows} and all(d.startswith("문서-") for d in docs)
    assert SH.build(assigned, seed=5) == (sheets, key, docs)


def test_cli_sheets_dev_writes_sheets_key_hash_and_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_VALIDATION_DIR", str(tmp_path / "검증"))
    jv = _jv()
    ledger = tmp_path / "장부.csv"
    jv.execute(jv.RunArgs(mode="dev", batch="reproducibility", ledger=ledger, out=tmp_path / "runs", yes=True, seed=1,
                          reports=["dev_binary", "dev_multiclass"], repeats=1), echo=lambda _: None)
    lines: list[str] = []
    draw = jv.cmd_sheets("dev", ledger=ledger, out=tmp_path / "사람", n=12, overlap=1, seed=9, echo=lines.append)
    out = tmp_path / "사람"
    assert draw["seed"] == 9 and draw["drawn"] <= 12 and (out / "뽑기.json").exists()
    assert draw["overlap"] == 1 and draw["overlap_drawn"] == 1 and sum(draw["overlap_by_axis"].values()) == 1
    key = (out / "열쇠" / "열쇠.csv").read_bytes()
    import hashlib

    assert (out / "열쇠" / "열쇠.sha256").read_text(encoding="utf-8").split()[0] == hashlib.sha256(key).hexdigest() == draw["key_sha256"]
    with (out / "채점지_인용.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [f"# 열쇠 sha256: {draw['key_sha256']}"]  # 머리 줄 — report 가 열쇠와 맞대 본다(J10)
    assert tuple(rows[1]) == SH.SHEET_COLUMNS and len(rows) > 2
    docs = sorted(p.name for p in (out / "문서").iterdir())
    assert docs and all(d.startswith("문서-") for d in docs)
    assert "dev_binary" not in "".join(docs)
    for d in docs:  # 문서 머리에도 바탕 태그가 없다(블라인드, J10)
        text = (out / "문서" / d).read_text(encoding="utf-8")
        assert text.startswith(f"# {d[:-4]}\n") and "dev_binary" not in text.splitlines()[0] and "바탕" not in text.splitlines()[0]
    assert "시드 9" in lines[0]
    assert "| 분석-1 |" in (out / "필수내용_항목.md").read_text(encoding="utf-8")


# ── 사람 채점 → 일치도 짝(sheets.human_pairs) ──


def _key_row(q, axis, system):
    return {"문항": q, "축": axis, "시스템_문제": system}


def _sheet_row(q, v):
    return {"문항": q, SH.SHEET_COLUMNS[6]: v}


def test_human_pairs_by_axis_with_overlap_and_blank():
    key = [_key_row("Q001", "faithfulness", "예"), _key_row("Q002", "coverage", "아니오"), _key_row("Q003", "coverage", "예")]
    filled = {"인용": [_sheet_row("Q001", "예"), _sheet_row("Q002", "예"), _sheet_row("Q003", "")],
              "민서": [_sheet_row("Q002", "아니오")]}
    got = SH.human_pairs(key, filled)
    assert got["blank"] == 1
    assert got["system_human"]["faithfulness"] == [("예", "예")]
    assert got["system_human"]["coverage"] == [("아니오", "아니오"), ("아니오", "예")]  # 민서·인용 순(이름 정렬)
    assert got["human_human"]["coverage"] == [("예", "아니오")] and got["human_human"]["faithfulness"] == []


def test_human_pairs_rejects_unknown_question_and_bad_verdict():
    key = [_key_row("Q001", "faithfulness", "예")]
    with pytest.raises(ValueError, match="열쇠에 없는 문항"):
        SH.human_pairs(key, {"인용": [_sheet_row("Q999", "예")]})
    with pytest.raises(ValueError, match="판정은 예/아니오"):
        SH.human_pairs(key, {"인용": [_sheet_row("Q001", "맞음")]})


def test_cli_sheets_defaults_are_forty_items_ten_overlap():
    """V1 결정 7(2026-10-29): sheets 기본 --n 40(축마다 10), --overlap 10(모두)."""
    jv = _jv()
    args = jv.build_parser().parse_args(["sheets", "--mode", "dev"])
    assert (args.n, args.overlap) == (40, 10) == (jv.SHEETS_N, jv.SHEETS_OVERLAP)
    import inspect

    sig = inspect.signature(jv.cmd_sheets).parameters
    assert (sig["n"].default, sig["overlap"].default) == (40, 10)
