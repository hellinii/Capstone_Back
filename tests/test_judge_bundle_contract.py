"""tests/test_judge_bundle_contract.py — 고객용 심판 묶음 계약 초안(심판 라운드 J1 ④, 계획 14-2).

`tests/contract/judge_bundle_ed4.json` 이 계약이다(프론트 src/test/contract/ 에 같은 바이트, 프론트
src/lib/report/judgeBundleContract.test.ts 가 같은 sha256 을 본다). 여기서는 계약 자체가 맞물리는지(예시가 칸 목록·보류 종류·
축 순서·필수 내용 18항목 분모와 맞는지) 본다. 백엔드가 이 모양을 실제로 내는지는 J5b 시험이 이 예시로 본다.
"""
import hashlib
import json
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
NAME = "judge_bundle_ed4.json"
PATH = TESTS / "contract" / NAME
FRONT_COPY = TESTS.parents[1] / "Capstone_Front" / "src" / "test" / "contract" / NAME
SHA256 = "e4bc794868a7bb8cc781765ae383c2de647328d434986a6cb245bdccb6d1c25e"
C = json.loads(PATH.read_text(encoding="utf-8"))
AXES = ["faithfulness", "consistency", "coverage", "readability"]
TOP = ["gate", "held_kind", "retry_allowed", "criteria_version", "model", "axes", "counts"]


def test_pinned_by_sha256():
    assert hashlib.sha256(PATH.read_bytes()).hexdigest() == SHA256


def test_fields_list_matches_examples():
    assert [k for k in C["fields"] if "." not in k] == TOP
    examples = C["examples"]
    assert set(examples) == {"passed", "held"}
    for ex in examples.values():
        assert list(ex) == TOP
        assert ex["axes"] == AXES


def test_passed_example_counts_cover_every_axis():
    ex = C["examples"]["passed"]
    assert (ex["gate"], ex["held_kind"], ex["retry_allowed"]) == ("passed", None, False)
    counts = ex["counts"]
    assert list(counts) == AXES
    assert set(counts["faithfulness"]) == {"checked", "matched"}
    assert set(counts["consistency"]) == {"checked", "contradictions"}
    cov = counts["coverage"]
    assert cov["applicable"] + cov["not_applicable"] == 18  # 필수 내용 18항목(설계 6.2)
    assert len(cov["items"]) == cov["applicable"] and cov["described"] == sum(i["described"] for i in cov["items"])
    assert all(set(i) == {"id", "described", "place"} for i in cov["items"])
    assert set(counts["readability"]) == {"rules", "violations"}


def test_held_example_has_kind_and_no_counts():
    ex = C["examples"]["held"]
    assert ex["gate"] == "held" and ex["held_kind"] in C["held_kinds"] and ex["counts"] is None


def test_held_kinds_fixed_list():
    assert list(C["held_kinds"]) == ["gate_failed", "timeout", "judge_error", "no_key", "budget", "job_limit", "no_material"]


@pytest.mark.skipif(not FRONT_COPY.exists(), reason="옆 저장소(Capstone_Front)가 없다 — CI 는 저장소 하나만 받는다")
def test_front_copy_is_byte_identical():
    assert FRONT_COPY.read_bytes() == PATH.read_bytes()
