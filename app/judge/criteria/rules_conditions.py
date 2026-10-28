"""criteria/rules_conditions.py — 필수 내용(Coverage) 적용 항목 고르기(설계 6.2 "항목 + 적용 조건", Q4 d).

조건 종류는 작업 유형과 값 유무 둘뿐이다. 요청 데이터(사실 시트·고객 입력·작업이 계산한 참조 범위)만 보고 코드가 고른다.
조건 글은 checklist.json conditions. 서술 쪽 판정 규칙(app.narrative.required_items·wording)을 import 하지 않고 필요한 작은 도구는
여기 복사한다(심판 기준이 서술 검문을 따라 몰래 바뀌지 않게). 같은 조건을 생성기도 따로 판단하는 곳(대표성 = 보고 항목 2)은
대조 표 tests/contract/reporting_item2_cases.json 으로 맞댄다.
"""
from __future__ import annotations

NOT_PROVIDED = "제공되지 않음"
EVAL_SOURCE_ROW = "평가 데이터 출처"
EVAL_RELATION_ROW = "학습 데이터와의 관계"


def is_not_provided(value) -> bool:
    """고객 입력 한 칸이 제공되지 않았는가 — 비었거나(글이 아님 포함) '제공되지 않음'으로 시작한다(서술 쪽 wording.is_not_provided 사본)."""
    return not isinstance(value, str) or not value.strip() or value.strip().startswith(NOT_PROVIDED)


def eval_data_source_provided(client_inputs) -> bool:
    """보고 항목 2 가 제공됨 — 출처와 학습 데이터와의 관계 두 줄 모두(재료 선택 점검 항목 2·서술 8.2-14 와 같은 규칙)."""
    test = client_inputs.get("test_data") if isinstance(client_inputs, dict) else None
    if not isinstance(test, dict):
        return False
    return not is_not_provided(test.get(EVAL_SOURCE_ROW)) and not is_not_provided(test.get(EVAL_RELATION_ROW))


def _judged(m) -> bool:
    return m.threshold is not None and m.value is not None and m.status in ("pass", "fail")


def conditions(task_type: str, fs, client_inputs, benchmark_refs: list[dict]) -> dict[str, bool]:
    facts = fs.facts
    declarations = getattr(facts, "declarations", None) if facts else None
    risk = getattr(declarations, "risk_direction", None) if declarations is not None else None
    if isinstance(declarations, dict):
        risk = declarations.get("risk_direction")
    shift_status = getattr(facts, "distribution_shift_status", None) if facts else None
    latency = getattr(fs, "latency", None)
    return {
        "always": True,
        "probability_present": fs.error_ratios is not None,
        "multi_and_two_average_rows": task_type in ("multiclass", "multilabel") and len(fs.class_averages or []) >= 2,
        "distribution_compared": shift_status == "compared",
        "binary_risk_declared": task_type == "binary" and risk in ("fn", "fp"),
        "judged_metric_present": any(_judged(m) for m in fs.metrics),
        "reference_present": bool(benchmark_refs),
        "latency_present": bool(latency is not None and getattr(latency, "available", False)),
        "eval_data_source_provided": eval_data_source_provided(client_inputs),
        "target_outside_reference": any(r.get("target_position") in ("below", "above") for r in benchmark_refs),
        "problems_present": len(fs.problems) > 0,
    }


def applicable_items(items: list[dict], task_type: str, fs, client_inputs, benchmark_refs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(적용 항목, 적용되지 않는 항목) — 체크리스트 순서 그대로."""
    cond = conditions(task_type, fs, client_inputs, benchmark_refs)
    on = [i for i in items if cond[i["condition"]]]
    off = [i for i in items if not cond[i["condition"]]]
    return on, off
