"""tests/test_judge_degree.py — 정도 표현 허용 판정(criteria/rules_degree.py, 설계 6.1·Q4 a)."""
from types import SimpleNamespace as M

import pytest

from app.judge.criteria.rules_degree import degree_verdict

ACC = M(metric_id="M1", display_name="Accuracy", value=0.95, threshold=0.9, status="pass", is_core=True)
REC = M(metric_id="M3", display_name="Recall", value=0.85, threshold=0.9, status="fail", is_core=True)
PREC = M(metric_id="M2", display_name="Precision", value=0.7, threshold=None, status="info", is_core=False)
NOVAL = M(metric_id="M9", display_name="AUC", value=None, threshold=0.9, status="unavailable", is_core=True)
REFS = [{"metric": "Accuracy", "quality": "within"}, {"metric": "Precision", "quality": "better"}]


@pytest.mark.parametrize("polarity, mid, quote, metrics, refs, want", [
    ("neutral", "", "적정한 수준", [REC], [], "ok"),
    ("positive", "M1", "양호하다", [ACC], [], "ok"),
    ("negative", "M1", "미흡하다", [ACC], [], "violation"),
    ("positive", "M3", "양호하다", [REC], [], "violation"),
    ("negative", "M3", "미흡하다", [REC], [], "ok"),
    ("positive", "M1", "우수하다", [ACC], REFS, "violation"),  # 강한 긍정 — 참조 범위 안이면 안 됨
    ("positive", "M1", "우수하다", [ACC], [], "ok"),            # 참조 범위 없으면 방향만
    ("positive", "M2", "높다", [PREC], REFS, "ok"),             # 목표 없는 지표 — 참조 범위보다 높음
    ("negative", "M2", "낮다", [PREC], REFS, "violation"),
    ("positive", "M2", "높다", [PREC], [], "target_unclear"),   # 목표도 참조 범위도 없음
    ("positive", "", "전반적으로 양호", [ACC, REC], [], "violation"),  # 핵심 지표 하나 미달
    ("negative", "", "전반적으로 미흡", [ACC, REC], [], "ok"),
    ("positive", "", "전반적으로 양호", [ACC], [], "ok"),
    ("negative", "", "전반적으로 미흡", [ACC], [], "violation"),
    ("positive", "", "양호", [NOVAL], [], "target_unclear"),      # 판정 불가 — 기준점 없음
    ("positive", "M9", "양호", [NOVAL], [], "target_unclear"),
    ("positive", "M7", "양호", [ACC], [], "target_unclear"),     # 없는 지표
    ("", "M1", "양호", [ACC], [], "target_unclear"),
])
def test_degree_table(polarity, mid, quote, metrics, refs, want):
    assert degree_verdict(polarity, mid, quote, metrics, refs) == want
