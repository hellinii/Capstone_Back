"""evaluation/metrics — task별 실제 지표 계산 함수(sklearn 기반).

engine 의 METRIC_REGISTRY 가 지표 별로 여기의 함수를 호출한다.
- common.py     : 세 task 공통 지표(정확도·정밀도·재현율·F1·혼동행렬 등)
- binary.py     : 이진 전용(AUROC/AUPRC/MCC·ROC/PR 곡선 등)
- multiclass.py : 다중클래스 전용(macro/micro/weighted 평균 등)
- multilabel.py : 다중레이블 전용(Hamming/Exact Match/Jaccard 등)
"""
from . import common
from . import binary
from . import multiclass
from . import multilabel

__all__ = ["common", "binary", "multiclass", "multilabel"]
