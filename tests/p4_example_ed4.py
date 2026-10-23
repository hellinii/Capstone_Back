"""tests/p4_example_ed4.py — 판 4 서술 시험(S2b ⑤⑥⑦⑨)이 같이 쓰는 입력과, 판 4 예시에 묶인 기대값을 **한 곳에** 모은 것.

판 4 예시 tests/contract/fact_sheet_binary_ed4.json 은 S2 ① 이 만든 **임시** 예시다(bc_strong — RPT-2026-0001 입력). S3 ⑧ 에서
프론트가 실제로 뜬 예시로 바뀐다. 그때 바뀔 수 있는 글자(규칙 문장·허용 목록의 몇 숫자)는 이 파일의 EXAMPLE_* 에만 적었다 —
다른 시험은 예시 글자에 기대지 않고 성질(판 4 표기·검문 통과·판 3 과 다름)로 본다. 예시를 바꿔 여기 값이 달라지면, 다른
시험이 아니라 이 파일만 이유와 함께 고친다(달라진 글자가 정말 맞는지 예시본 5·6절과 대조).

- edition4_cases(): 판 4 로 돌려 볼 요청들 — 판 4 예시 + 판 3 입력(계약 3종·얼린 합성 2종·발급본 요청 6건)에 report_edition 4
  만 붙인 것. 판 3 입력은 판 4 칸이 없어(None) 판 4 갈래의 '없을 때' 경로를 지난다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from app.narrative.schemas import NarrativeRequest

TESTS = Path(__file__).parent
CONTRACT = TESTS / "contract"
FROZEN_INPUTS = TESTS / "golden_frozen" / "inputs"
ISSUED_REQUESTS = TESTS / "data" / "edition3_issued" / "requests"
EXAMPLE_PATH = CONTRACT / "fact_sheet_binary_ed4.json"


def example_raw() -> dict:
    """판 4 예시 요청 원문(사본 — 고쳐도 파일은 그대로)."""
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def example_req() -> NarrativeRequest:
    return NarrativeRequest.model_validate(example_raw())


def as_edition(raw: dict, edition: int) -> dict:
    out = copy.deepcopy(raw)
    out["report_edition"] = edition
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def edition4_cases() -> list[tuple[str, dict]]:
    """(이름, 판 4 요청 원문) — 판 4 예시와, 판 3 입력에 판 신호만 4 로 붙인 것."""
    cases = [("ed4_example", example_raw())]
    for name in ("binary", "multiclass", "multilabel", "synth_inconclusive", "synth_conditional"):
        cases.append((f"ed3in_{name}", as_edition(_load(FROZEN_INPUTS / f"fact_sheet_{name}.json"), 4)))
    for path in sorted(ISSUED_REQUESTS.glob("*.request.json")):
        cases.append((f"issued_{path.name.split('.')[0]}", as_edition(_load(path), 4)))
    assert len(cases) == 12
    return cases


# ── 판 4 예시에 묶인 기대값(예시를 바꾸면 여기만 고친다) ─────────────────────────────────

# 규칙 문장(키 없음 — generate_sections(None, req))의 칸 글. 예시본 5·6절과 같은 판 4 표기·판 4 자리 글이다.
EXAMPLE_TEMPLATES = {
    "interpretation.confusion_analysis": (
        "혼동행렬 기준 전체 171건 중 3건(1.8%)이 오분류되었다. 양성 클래스 malignant(악성 종양) 기준으로 거짓 양성(FP)은 2건, "
        "거짓 음성(FN)은 1건이다. 의뢰자가 더 위험하다고 밝힌 거짓 음성(FN, 놓침)은 1건으로 전체 오류 3건의 33.3%이다. 양성 "
        "클래스 malignant(악성 종양)로 예측한 것은 65건, 실제 양성은 64건으로, 양성을 실제보다 1건 많이 예측했다. F1 이 가장 "
        "낮은 클래스는 F1 0.977인 malignant(악성 종양)이며, 표본은 64건으로 전체 표본의 37.4%이다. 매크로 F1 과의 차이는 "
        "0.005이다. 확률이 있는 오분류 가운데 경계 오류가 확신 오류보다 많다. 경계 오류는 결정 경계 가까이에서 갈린 오류로, "
        "경계에 가까운 사례의 판단에 따라 결과가 달라질 수 있는 오류다."
    ),
    "interpretation.distribution_analysis": (
        "평가 데이터셋은 총 171건이며, 클래스 분포는 benign(양성 종양) 107건(62.6%), malignant(악성 종양) 64건(37.4%)이다. "
        "불균형 비율(1.672)이 1.5 이상이어서 불균형이 있으나, 불균형에 의한 성능 왜곡은 관찰되지 않았다. 정확도와 매크로 F1 의 "
        "차이(0.001)는 기준(0.1) 미만이다. 다수 클래스(benign(양성 종양))와 소수 클래스(malignant(악성 종양))의 F1 "
        "차이(0.009)는 기준(0.1) 미만이다. 학습 분포와 평가 분포의 클래스 비율 차는 모두 10%p 미만이다."
    ),
    "interpretation.additional_observations": "",
    "conclusion.benchmark": (
        "내부 참조 기준과 비교하면 참조 범위보다 높은 지표는 Accuracy(0.982, 참조 범위 0.85~0.92), Precision(0.969, 참조 범위 "
        "0.80~0.92), Recall(0.984, 참조 범위 0.80~0.92), F1 Score(0.977, 참조 범위 0.83~0.91), AUROC(0.999, 참조 범위 "
        "0.85~0.95), AUPRC(0.999, 참조 범위 0.80~0.93)이다. 비교 기준은 내부 참조 기준치 v1 (공개 벤치마크 평균이 아닌 자체 설정 "
        "기준이며, 도메인별로 상이할 수 있음)이다. 목표값이 참조 범위보다 높은 지표는 Accuracy(목표 0.950), Recall(목표 0.970), "
        "F1 Score(목표 0.950), AUROC(목표 0.980)이다."
    ),
    "conclusion.narrative": (
        "목표값이 설정된 판정 대상 지표가 모두 목표를 충족하여 종합 판정은 '최종 합격'이다. 참고로 목표값이 설정된 지표 5개 중 "
        "5개가 목표를 충족하였다. 95% 신뢰구간이 목표값을 포함하는 지표는 Accuracy(구간 0.9497~0.994, 목표 0.950), Recall(구간 "
        "0.917~0.997, 목표 0.970), F1 Score(구간 0.946~1.000, 목표 0.950)로, 표본 변동에 따라 해당 지표의 판정이 달라질 수 있다. "
        "그 밖에 신뢰구간을 산출한 지표(AUROC, Log Loss)는 구간이 목표값과 떨어져 있다. 양성 클래스의 Precision 은 0.969, "
        "Recall 은 0.984로 Recall 쪽이 0.015 높고, 두 값의 차이는 기준(0.1) 미만이어서 치우침은 크지 않다. 의뢰자가 밝힌 모델 "
        "용도는 '유방 종괴 세침흡인(FNA) 영상에서 뽑은 세포핵 특징으로 악성 여부를 가려, 판독 대기열의 우선순위를 정하는 데 "
        "쓴다.'이다."
    ),
    "conclusion.risks": (
        "목표값이 설정된 모든 시험항목이 목표를 충족하였다. F1 이 가장 낮은 클래스는 F1 0.977인 malignant(악성 종양)이다. 95% "
        "신뢰구간이 목표값을 포함하는 Accuracy, Recall, F1 Score는 판정의 여유가 작아, 표본이 달라지면 판정이 뒤집힐 수 있다."
    ),
    "recommendation_narrative.data_quality": (
        "데이터 측면에서 지목된 문제는 P2(평가 데이터가 학습과 같은 출처)이며, 6절 권고표의 해당 항목에 따른 보완 후 재평가를 "
        "권고한다."
    ),
    "recommendation_narrative.model_ops": (
        "모델 측면에서 지목된 문제는 P1(핵심 지표 근소 합격 (Recall))이며, 6절 권고표의 해당 항목에 따른 개선 후 재평가를 권고한다."
    ),
}

# 판 4 허용 목록에 들어야 하는 표기(판 4 보기·규칙 문장의 숫자 — 늘린 표기 0.9497·0.6166 포함)와, 들면 안 되는 넷째 자리
# 표기(판 3 허용 목록에는 있다 — 시험이 함께 단정해 '우연히 없는 숫자'가 아님을 보인다). 통과율 100 도 판 4 에서는 뺀다.
EXAMPLE_ALLOWED = ("0.9497", "0.6166", "0.977", "0.950", "0.984", "0.969", "1.672", "0.005", "0.008", "37.4%", "1.8%", "33.3%",
                   "−0.2%p", "+0.2%p", "0.80", "0.92", "0.5", "171", "64")
EXAMPLE_FOURTH_DIGIT = ("0.9767", "0.9825", "0.9692", "0.9844", "0.0076", "0.0325", "0.0046", "0.3743", "1.6719", "0.0092")
EXAMPLE_PASS_RATE = "100"

# 위험 쪽 지표 문장(부록 B20)을 붙인 8.2-06 — 예시의 상승폭을 0.10 이상으로 바꾼 사본(rules 시험의 _tuning)에서.
EXAMPLE_B20_FN = (
    "의뢰자가 더 위험하다고 밝힌 거짓 음성(FN) 쪽 지표인 Recall은 F1이 가장 높은 결정 임계값 0.6166에서 0.984이다. "
    "이 값은 목표 0.970을 충족한다."
)
