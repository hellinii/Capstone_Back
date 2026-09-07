"""지표 값의 정확성을 손검산 기대값으로 고정하는 스위트.

ISSUES.md H-01 — 기존 `test_evaluator.py:assert_valid_results()` 의 단정은
`val is not None` / `not isnan` / `0.0 <= val <= 1.0` 셋뿐이고 픽스처가 전부 무작위
노이즈(binary accuracy 0.505, corr(score, y_true)=0.02)라, **공식이 틀려도 값이
그럴듯해서 통과한다.** 골든 스냅샷도 그 출력의 복사본이라 오라클이 아니다.

`test_metric_edge_cases.py` 가 경계 조건(C-01·C-02·A-07·D-05·D-07·D-10)에 대한
기대값을 이미 고정했지만, 그중 비퇴화 기대값을 가진 지표는 M7·M8·M17 셋뿐이었다
(M1·M15·M16 의 단정은 완전예측 데이터에 대한 값이라 공식이 틀려도 통과한다).

이 파일은 **작고 손으로 검산 가능한 픽스처**로 지표별 기대값을 못박는다. 각 기대값은
주석에 유도 과정을 남긴다 — 값을 바꾸려면 유도부터 반박해야 한다.

평균 규약(A-08 결정, 2026-09-05): M2~M5 는 binary 를 제외한 전 task 에서 **macro**,
M17 만 samples. 이 파일의 multilabel 기대값이 그 규약을 코드로 고정한다.
"""
import pandas as pd
import pytest

from app.evaluation.metrics import binary, common, multiclass, multilabel


# ══════════════════════════════════════════════════════════════════════════════
# 픽스처 — 전부 손으로 검산 가능한 크기
# ══════════════════════════════════════════════════════════════════════════════

def _mc_df():
    """multiclass 9행. 클래스 지지도를 A=4·B=3·C=2 로 다르게 둬 macro 와 weighted 를 구분한다.

        행 1 2 3 4 5 6 7 8 9
        t  A A A A B B B C C
        p  A A A B B B C C C      → 맞은 행 7개(4·7행만 틀림)

    혼동행렬(행=정답, 열=예측, 라벨 [A,B,C]):
        A: [3, 1, 0]      B: [0, 2, 1]      C: [0, 0, 2]
    클래스별: A P=3/3, R=3/4, F1=6/7 · B P=2/3, R=2/3, F1=2/3 · C P=2/3, R=2/2, F1=4/5
    """
    return pd.DataFrame({
        "t": ["A", "A", "A", "A", "B", "B", "B", "C", "C"],
        "p": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    })


def _mc_mapping():
    return {"y_true": "t", "y_pred": "p", "_task_type": "multiclass", "_beta": 1.0}


def _bin_df():
    """binary 8행. P≠R 이 되도록 배치해 F1·Fbeta 가 P·R 로 퇴화하지 않게 한다.

        t  1 1 1 1 0 0 0 0
        p  1 1 0 0 0 0 0 1      → TP=2, FN=2, TN=3, FP=1
    """
    return pd.DataFrame({
        "t": [1, 1, 1, 1, 0, 0, 0, 0],
        "p": [1, 1, 0, 0, 0, 0, 0, 1],
    })


def _bin_mapping(beta=1.0):
    return {"y_true": "t", "y_pred": "p", "_task_type": "binary",
            "_positive_class": 1, "_beta": beta}


def _ml_df():
    """multilabel 4행 / 라벨 {a,b,c}. macro 와 samples 가 뚜렷이 갈리도록 배치했다.

        행  true      pred        이진화(열 순서 a,b,c)
        1   [a,b]     [a,b]       t=110  p=110
        2   [a]       [a,b]       t=100  p=110
        3   [b,c]     [b]         t=011  p=010
        4   [c]       [a]         t=001  p=100

    라벨별 열: a t=1100 p=1101 · b t=1010 p=1110 · c t=0011 p=0000
    """
    return pd.DataFrame({
        "t": [["a", "b"], ["a"], ["b", "c"], ["c"]],
        "p": [["a", "b"], ["a", "b"], ["b"], ["a"]],
    })


def _ml_mapping():
    return {"true_labels": "t", "pred_labels": "p", "_task_type": "multilabel", "_beta": 1.0}


# ══════════════════════════════════════════════════════════════════════════════
# M1 Accuracy
# ══════════════════════════════════════════════════════════════════════════════

def test_m1_accuracy_multiclass():
    """[H-01] 9행 중 7행이 맞았다 → 7/9."""
    assert common.calculate_accuracy(_mc_df(), _mc_mapping()) == pytest.approx(7 / 9)


def test_m1_accuracy_binary():
    """[H-01] TP=2, TN=3 → (2+3)/8."""
    assert common.calculate_accuracy(_bin_df(), _bin_mapping()) == pytest.approx(5 / 8)


def test_m1_accuracy_multilabel_is_subset_accuracy():
    """[H-01] multilabel 에서 M1 은 '행 전체가 정확히 일치' 만 센다 → 1행/4행.

    이 값이 M16(Exact Match Ratio)과 같아야 한다는 것이 M1 ≡ M16 의 내용이다
    (app/core/schemas.py 가 동일값임을 이미 명시).
    """
    assert common.calculate_accuracy(_ml_df(), _ml_mapping()) == pytest.approx(0.25)
    assert multilabel.calculate_exact_match_ratio(_ml_df(), _ml_mapping()) == pytest.approx(0.25)


# ══════════════════════════════════════════════════════════════════════════════
# M2~M5 Precision / Recall / F1 / Fbeta — binary
# ══════════════════════════════════════════════════════════════════════════════

def test_m2_m3_m4_binary():
    """[H-01] TP=2, FP=1, FN=2 → P=2/3, R=1/2, F1=2PR/(P+R)=4/7."""
    df, m = _bin_df(), _bin_mapping()
    assert common.calculate_precision(df, m) == pytest.approx(2 / 3)
    assert common.calculate_recall(df, m) == pytest.approx(1 / 2)
    assert common.calculate_f1_score(df, m) == pytest.approx(4 / 7)


def test_m5_fbeta_binary_weights_recall_when_beta_is_two():
    """[H-01] beta=2 는 recall 을 더 무겁게 본다.

    F2 = 5PR / (4P + R) = 5·(2/3)·(1/2) / (4·(2/3) + 1/2) = (5/3)/(19/6) = 10/19.
    R(0.5) < P(2/3) 이므로 F2(0.526) < F1(0.571) 이어야 한다 — beta 가 실제로
    반영되지 않으면 이 부등식이 깨진다.
    """
    df = _bin_df()
    f1 = common.calculate_f1_score(df, _bin_mapping(beta=1.0))
    f2 = common.calculate_fbeta_score(df, _bin_mapping(beta=2.0))
    assert f2 == pytest.approx(10 / 19)
    assert f2 < f1


def test_m5_fbeta_beta_half_weights_precision():
    """[H-01] beta=0.5 는 precision 쪽으로 기운다 → F0.5 > F1.

    F0.5 = 1.25PR / (0.25P + R) = 1.25·(1/3) / (1/6 + 1/2) = (5/12)/(2/3) = 5/8.
    """
    assert common.calculate_fbeta_score(_bin_df(), _bin_mapping(beta=0.5)) == pytest.approx(5 / 8)


# ══════════════════════════════════════════════════════════════════════════════
# M2~M4 — multiclass macro (A-08 결정: macro 가 규약)
# ══════════════════════════════════════════════════════════════════════════════

def test_m2_m3_m4_multiclass_use_macro_average():
    """[H-01][A-08] 클래스별 값의 단순평균(macro)이다. 지지도 가중이 아니다.

    P = (3/3 + 2/3 + 2/3)/3 = 7/9
    R = (3/4 + 2/3 + 2/2)/3 = 29/36
    F1 = (6/7 + 2/3 + 4/5)/3 = 244/315
    """
    df, m = _mc_df(), _mc_mapping()
    assert common.calculate_precision(df, m) == pytest.approx(7 / 9)
    assert common.calculate_recall(df, m) == pytest.approx(29 / 36)
    assert common.calculate_f1_score(df, m) == pytest.approx(244 / 315)


def test_multiclass_macro_differs_from_weighted():
    """[H-01] 지지도가 다르므로 macro 와 weighted 는 달라야 한다 — 둘을 혼동하면 걸린다."""
    df, m = _mc_df(), _mc_mapping()
    assert common.calculate_precision(df, m) != pytest.approx(
        multiclass.calculate_weighted_average(df, m)["precision"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# M2~M5 — multilabel macro (A-08 결정을 코드로 고정)
# ══════════════════════════════════════════════════════════════════════════════

def test_m2_m3_m4_multilabel_use_macro_not_samples():
    """[H-01][A-08] 레이블별 값의 단순평균(macro)이다. 샘플별 평균(samples)이 아니다.

    레이블별 — a: TP=2 FP=1 FN=0 → P=2/3, R=1, F1=4/5
               b: TP=2 FP=1 FN=0 → P=2/3, R=1, F1=4/5
               c: TP=0 FP=0 FN=2 → P=0,   R=0, F1=0
    macro P = (2/3+2/3+0)/3 = 4/9 · R = (1+1+0)/3 = 2/3 · F1 = (4/5+4/5+0)/3 = 8/15

    대조 — samples 평균이면 P=R=0.625 가 나온다(행별 1, 0.5, 1, 0 / 1, 1, 0.5, 0).
    두 규약이 뚜렷이 갈리므로 이 테스트가 A-08 의 결정을 지킨다.
    """
    df, m = _ml_df(), _ml_mapping()
    assert common.calculate_precision(df, m) == pytest.approx(4 / 9)
    assert common.calculate_recall(df, m) == pytest.approx(2 / 3)
    assert common.calculate_f1_score(df, m) == pytest.approx(8 / 15)
    # samples 규약이었다면 나왔을 값과 실제로 다른지 못박는다
    assert common.calculate_precision(df, m) != pytest.approx(0.625)


# ══════════════════════════════════════════════════════════════════════════════
# M11~M13 macro / micro / weighted
# ══════════════════════════════════════════════════════════════════════════════

def test_m11_macro_average_matches_individual_metrics():
    """[H-01] M11 의 세 값은 M2·M3·M4(macro)와 같아야 한다."""
    df, m = _mc_df(), _mc_mapping()
    out = multiclass.calculate_macro_average(df, m)
    assert out["precision"] == pytest.approx(7 / 9)
    assert out["recall"] == pytest.approx(29 / 36)
    assert out["f1_score"] == pytest.approx(244 / 315)


def test_m12_micro_average_equals_accuracy_for_single_label():
    """[H-01] M12: 단일 레이블 multiclass 에서 micro P=R=F1=accuracy=7/9 다.

    micro 는 전 클래스의 TP/FP/FN 을 합산하므로, 한 행에 정답이 하나뿐이면
    ∑TP=맞은 행 수, ∑FP=∑FN=틀린 행 수가 되어 셋이 모두 accuracy 로 수렴한다.
    """
    out = multiclass.calculate_micro_average(_mc_df(), _mc_mapping())
    assert out["precision"] == pytest.approx(7 / 9)
    assert out["recall"] == pytest.approx(7 / 9)
    assert out["f1_score"] == pytest.approx(7 / 9)


def test_m13_weighted_average_uses_support_weights():
    """[H-01] 지지도(A=4, B=3, C=2)로 가중평균한다.

    P = (4·1 + 3·2/3 + 2·2/3)/9 = 22/27
    R = (4·3/4 + 3·2/3 + 2·1)/9 = 7/9   ← weighted recall 은 accuracy 와 같다
    F1 = (4·6/7 + 3·2/3 + 2·4/5)/9 = 82/105
    """
    out = multiclass.calculate_weighted_average(_mc_df(), _mc_mapping())
    assert out["precision"] == pytest.approx(22 / 27)
    assert out["recall"] == pytest.approx(7 / 9)
    assert out["f1_score"] == pytest.approx(82 / 105)


# ══════════════════════════════════════════════════════════════════════════════
# M14 / M18 분포 차이
# ══════════════════════════════════════════════════════════════════════════════

def test_m14_distribution_diff_multiclass_is_tvd():
    """[H-01] 정답 분포 (4,3,2)/9 vs 예측 분포 (3,3,3)/9 의 TVD.

    TVD = ½·(|4/9-3/9| + |3/9-3/9| + |2/9-3/9|) = ½·(2/9) = 1/9
    """
    assert multiclass.calculate_distribution_diff_mc(_mc_df(), _mc_mapping()) == pytest.approx(1 / 9)


def test_m14_is_zero_when_distributions_match_despite_wrong_predictions():
    """[H-01] TVD 는 분포만 본다 — 전부 틀려도 분포가 같으면 0 이다.

    이 성질을 모르면 M14 를 정확도류로 오해한다. 성적서 해석에 직결된다.
    """
    df = pd.DataFrame({"t": ["A", "B"], "p": ["B", "A"]})
    assert multiclass.calculate_distribution_diff_mc(df, _mc_mapping()) == pytest.approx(0.0)


def test_m18_distribution_diff_multilabel_is_cosine_distance():
    """[H-01][A-05] 레이블 빈도 벡터의 코사인 거리다. TVD 가 아니다.

    p_freq = (a2, b2, c2), q_freq = (a3, b3, c0)
    cos = 12 / (√12 · √18) = 2/√6  →  거리 = 1 - 2/√6 ≈ 0.1835034
    """
    expected = 1.0 - 12 / ((12 ** 0.5) * (18 ** 0.5))
    assert multilabel.calculate_distribution_diff_ml(_ml_df(), _ml_mapping()) == pytest.approx(expected)
    assert expected == pytest.approx(0.1835034, abs=1e-7)


# ══════════════════════════════════════════════════════════════════════════════
# M15 / M16 / M17 multilabel
# ══════════════════════════════════════════════════════════════════════════════

def test_m15_hamming_loss_counts_wrong_cells():
    """[H-01] 어긋난 셀 수 / (행 4 × 라벨 3).

    행1 0개 · 행2 b 1개 · 행3 c 1개 · 행4 a·c 2개 → 4/12 = 1/3
    """
    assert multilabel.calculate_hamming_loss(_ml_df(), _ml_mapping()) == pytest.approx(1 / 3)


def test_m16_exact_match_ratio_counts_whole_rows():
    """[H-01] 행 전체가 일치한 것은 1행뿐 → 1/4."""
    assert multilabel.calculate_exact_match_ratio(_ml_df(), _ml_mapping()) == pytest.approx(0.25)


def test_m17_jaccard_is_sample_averaged():
    """[H-01] 행별 |교집합|/|합집합| 의 평균.

    행1 2/2=1 · 행2 1/2 · 행3 1/2 · 행4 0/2=0 → (1+0.5+0.5+0)/4 = 0.5
    """
    assert multilabel.calculate_jaccard_index(_ml_df(), _ml_mapping()) == pytest.approx(0.5)


def test_m15_m16_m17_disagree_on_the_same_data():
    """[H-01] 세 지표는 서로 다른 것을 측정한다 — 값이 같게 나오면 구현이 섞인 것이다."""
    df, m = _ml_df(), _ml_mapping()
    h = multilabel.calculate_hamming_loss(df, m)
    e = multilabel.calculate_exact_match_ratio(df, m)
    j = multilabel.calculate_jaccard_index(df, m)
    assert len({round(h, 6), round(e, 6), round(j, 6)}) == 3


# ══════════════════════════════════════════════════════════════════════════════
# M20 MCC
# ══════════════════════════════════════════════════════════════════════════════

def test_m20_mcc_expected_value():
    """[H-01] MCC = (TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)).

    TP=2, TN=3, FP=1, FN=2 → (6−2)/√(3·4·4·5) = 4/√240
    """
    assert binary.calculate_mcc(_bin_df(), _bin_mapping()) == pytest.approx(4 / (240 ** 0.5))


def test_m20_mcc_is_zero_for_no_better_than_chance():
    """[H-01] 예측이 정답과 무상관이면 0 이다.

    t=[1,1,0,0], p=[1,0,1,0] → TP=1,FP=1,FN=1,TN=1 → 분자 0
    """
    df = pd.DataFrame({"t": [1, 1, 0, 0], "p": [1, 0, 1, 0]})
    assert binary.calculate_mcc(df, _bin_mapping()) == pytest.approx(0.0)


def test_m20_mcc_is_negative_when_predictions_are_inverted():
    """[H-01] 완전히 뒤집힌 예측은 −1 이다. [0,1] 범위 단정만으로는 잡히지 않는 성질."""
    df = pd.DataFrame({"t": [1, 1, 0, 0], "p": [0, 0, 1, 1]})
    assert binary.calculate_mcc(df, _bin_mapping()) == pytest.approx(-1.0)


# ══════════════════════════════════════════════════════════════════════════════
# M21 혼동행렬 / M22 클래스별 지표
# ══════════════════════════════════════════════════════════════════════════════

def test_m21_confusion_matrix_multiclass_layout():
    """[H-01] 행=정답, 열=예측, 라벨은 정렬 순서 [A,B,C]."""
    out = common.calculate_confusion_matrix(_mc_df(), _mc_mapping())
    assert out["type"] == "multiclass_or_binary"
    assert out["labels"] == ["A", "B", "C"]
    assert out["matrix"] == [[3, 1, 0], [0, 2, 1], [0, 0, 2]]


def test_m21_confusion_matrix_multilabel_is_per_label_2x2():
    """[H-01] multilabel 은 라벨마다 [[TN,FP],[FN,TP]] 를 준다.

    라벨 a: t=1100 p=1101 → TN=1, FP=1, FN=0, TP=2
    라벨 c: t=0011 p=0000 → TN=2, FP=0, FN=2, TP=0
    """
    out = common.calculate_confusion_matrix(_ml_df(), _ml_mapping())
    assert out["type"] == "multilabel"
    assert out["labels"] == ["a", "b", "c"]
    assert out["matrix"][0] == [[1, 1], [0, 2]]
    assert out["matrix"][2] == [[2, 0], [2, 0]]


def test_m22_class_metrics_per_class_values():
    """[H-01] classification_report 의 클래스별 값이 손계산과 일치한다."""
    rep = common.calculate_class_metrics(_mc_df(), _mc_mapping())
    assert rep["A"]["precision"] == pytest.approx(1.0)
    assert rep["A"]["recall"] == pytest.approx(3 / 4)
    assert rep["A"]["support"] == pytest.approx(4)
    assert rep["B"]["precision"] == pytest.approx(2 / 3)
    assert rep["C"]["recall"] == pytest.approx(1.0)
    assert rep["accuracy"] == pytest.approx(7 / 9)


def test_m22_macro_avg_matches_m11():
    """[H-01] M22 의 'macro avg' 는 M11 과 같은 수여야 한다.

    성적서는 M22 를 클래스별 표로 인쇄하고 M2~M4 를 헤드라인으로 인쇄한다.
    독자가 표를 평균해 헤드라인을 재계산할 수 있어야 한다(A-08 을 macro 로 둔 근거).
    """
    df, m = _mc_df(), _mc_mapping()
    rep = common.calculate_class_metrics(df, m)
    assert rep["macro avg"]["precision"] == pytest.approx(common.calculate_precision(df, m))
    assert rep["macro avg"]["recall"] == pytest.approx(common.calculate_recall(df, m))


# ══════════════════════════════════════════════════════════════════════════════
# M23 Imbalance Ratio
# ══════════════════════════════════════════════════════════════════════════════

def test_m23_imbalance_ratio_multiclass():
    """[H-01] 최다 클래스 A=4 / 최소 클래스 C=2 → 2.0. 예측은 보지 않는다."""
    assert common.calculate_imbalance_ratio(_mc_df(), _mc_mapping()) == pytest.approx(2.0)


def test_m23_ignores_predictions():
    """[H-01] M23 은 데이터셋 특성 지표다 — y_pred 를 바꿔도 값이 변하면 안 된다."""
    df = _mc_df()
    before = common.calculate_imbalance_ratio(df, _mc_mapping())
    df["p"] = ["A"] * 9
    assert common.calculate_imbalance_ratio(df, _mc_mapping()) == pytest.approx(before)


def test_m23_multilabel_counts_label_occurrences():
    """[H-01] multilabel 은 레이블 등장 횟수로 센다.

    true = [a,b], [a], [a] → a 3회, b 1회 → 3.0
    """
    df = pd.DataFrame({
        "t": [["a", "b"], ["a"], ["a"]],
        "p": [["a"], ["a"], ["a"]],
    })
    assert common.calculate_imbalance_ratio(df, _ml_mapping()) == pytest.approx(3.0)


def test_m23_balanced_dataset_is_one():
    """[H-01] 완전 균형이면 1.0 — 값이 클수록 불균형이라는 방향성을 고정한다."""
    df = pd.DataFrame({"t": ["A", "A", "B", "B"], "p": ["A", "A", "B", "B"]})
    assert common.calculate_imbalance_ratio(df, _mc_mapping()) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# M7 / M8 — 비퇴화 기대값 보강 (기존 edge-case 스위트는 경계 조건만 다룬다)
# ══════════════════════════════════════════════════════════════════════════════

def test_m7_m8_specificity_and_fpr_are_complementary():
    """[H-01] TN=3, FP=1 → Specificity 3/4, FPR 1/4. 둘의 합은 항상 1 이다."""
    df, m = _bin_df(), _bin_mapping()
    spec = binary.calculate_specificity(df, m)
    fpr = binary.calculate_fpr(df, m)
    assert spec == pytest.approx(3 / 4)
    assert fpr == pytest.approx(1 / 4)
    assert spec + fpr == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 확률 기반 지표 4개 — M6·M9·M10·M19 (ISSUES.md H-01 의 마지막 잔여)
#
# 이 넷은 "근사 유도가 필요하다"는 이유로 앞선 라운드에서 미착수로 남았다. 그러나 넷 다
# 작은 픽스처에서는 손으로 끝까지 계산된다. 남겨 두면 값 검증이 되는 지표가 19/23 에
# 머무르고, 특히 M9·M10·M19 는 **확률 전용 경로(결정 1)의 핵심 지표**라 이번 라운드에서
# 회귀 위험이 가장 큰 자리다.
# ══════════════════════════════════════════════════════════════════════════════

def _score_df():
    """binary 4행. 양성 하나가 음성 하나보다 낮은 점수를 받게 해 AUROC 를 비퇴화로 만든다.

        y_true : 1     1     0     0
        score  : 0.9   0.3   0.35  0.1

    양성 {0.9, 0.3}, 음성 {0.35, 0.1}.
    """
    return pd.DataFrame({
        "y_true": [1, 1, 0, 0],
        "score": [0.9, 0.3, 0.35, 0.1],
    })


_SCORE_MAP = {"y_true": "y_true", "score_positive": "score", "_positive_class": 1,
              "_task_type": "binary"}


def test_m9_auroc_is_the_share_of_correctly_ordered_pairs():
    """AUROC = (양성, 음성) 쌍 중 양성 점수가 더 높은 쌍의 비율.

    쌍 4개: (0.9,0.35)✓ (0.9,0.1)✓ (0.3,0.35)✗ (0.3,0.1)✓ → 3/4 = 0.75.
    무작위 노이즈 픽스처에서는 0.5 근처가 나와 공식이 틀려도 그럴듯하다 —
    여기서는 0.75 라는 특정 값이 어긋나면 곧바로 드러난다.
    """
    assert binary.calculate_auroc(_score_df(), _SCORE_MAP) == pytest.approx(0.75)


def test_m9_auroc_is_one_for_perfect_ranking():
    df = pd.DataFrame({"y_true": [1, 1, 0, 0], "score": [0.9, 0.8, 0.2, 0.1]})
    assert binary.calculate_auroc(df, _SCORE_MAP) == pytest.approx(1.0)


def test_m9_auroc_is_zero_for_inverted_ranking():
    """순위를 뒤집으면 0 이다 — 0.5(무작위)와 구분되는지 확인한다."""
    df = pd.DataFrame({"y_true": [1, 1, 0, 0], "score": [0.1, 0.2, 0.8, 0.9]})
    assert binary.calculate_auroc(df, _SCORE_MAP) == pytest.approx(0.0)


def test_m10_auprc_is_the_step_wise_precision_recall_area():
    """AP = Σ (R_n − R_{n−1}) · P_n. 점수 내림차순으로 훑는다.

        0.9  (양성) → TP1 FP0 → P=1,   R=1/2   ΔR=1/2 → 1 × 1/2   = 1/2
        0.35 (음성) → TP1 FP1 → P=1/2, R=1/2   ΔR=0   → 0
        0.3  (양성) → TP2 FP1 → P=2/3, R=1     ΔR=1/2 → 2/3 × 1/2 = 1/3
        0.1  (음성) → TP2 FP2 → P=1/2, R=1     ΔR=0   → 0
        AP = 1/2 + 1/3 = 5/6
    """
    assert binary.calculate_auprc(_score_df(), _SCORE_MAP) == pytest.approx(5 / 6)


def test_m10_auprc_baseline_is_the_positive_rate_when_scores_carry_no_signal():
    """점수가 정보를 주지 못하면 AP 는 양성 비율로 수렴한다 — 0.5 가 아니다.

    AUROC 의 무정보 기준선(0.5)과 다르다는 사실을 고정한다. 둘을 혼동한 구현은
    이 단정에서 걸린다.
    """
    df = pd.DataFrame({"y_true": [1, 0, 0, 0], "score": [0.5, 0.5, 0.5, 0.5]})
    assert binary.calculate_auprc(df, _SCORE_MAP) == pytest.approx(0.25)


def test_m19_log_loss_expected_value():
    """LogLoss = −(1/N) Σ [y·ln(p) + (1−y)·ln(1−p)].

        −(1/4)[ ln(0.9) + ln(0.3) + ln(0.65) + ln(0.9) ]
        = −(1/4)[ −0.1053605 − 1.2039728 − 0.4307829 − 0.1053605 ]
        = 1.8454767 / 4 = 0.4613692
    """
    assert binary.calculate_log_loss(_score_df(), _SCORE_MAP) == pytest.approx(0.4613692, abs=1e-6)


def test_m19_log_loss_grows_when_a_confident_prediction_is_wrong():
    """자신 있게 틀린 예측이 더 크게 벌점을 받는다 — 방향성을 고정한다."""
    mild = pd.DataFrame({"y_true": [1, 0], "score": [0.6, 0.4]})
    confident_wrong = pd.DataFrame({"y_true": [1, 0], "score": [0.1, 0.9]})

    assert binary.calculate_log_loss(confident_wrong, _SCORE_MAP) > binary.calculate_log_loss(mild, _SCORE_MAP)


def test_m6_kl_divergence_between_label_distributions():
    """M6 는 확률이 아니라 **정답/예측 라벨의 빈도 분포** 사이의 KL 이다(A-09·SPEC §2 규칙 3).

        y_true = A A B C  →  p = {A:1/2, B:1/4, C:1/4}
        y_pred = A B B C  →  q = {A:1/4, B:1/2, C:1/4}

        KL(p‖q) = 1/2·ln(2) + 1/4·ln(1/2) + 1/4·ln(1)
                = 1/4·ln(2) = 0.1732868
    """
    df = pd.DataFrame({"y_true": ["A", "A", "B", "C"], "y_pred": ["A", "B", "B", "C"]})
    mapping = {"y_true": "y_true", "y_pred": "y_pred", "_task_type": "multiclass"}

    assert common.calculate_kl_divergence(df, mapping) == pytest.approx(0.1732868, abs=1e-6)


def test_m6_kl_divergence_is_zero_when_distributions_match():
    """분포가 같으면 0 이다 — **예측이 전부 틀려도** 그렇다(M14 와 같은 성질).

    KL 을 정확도처럼 구현한 코드는 이 단정에서 걸린다.
    """
    df = pd.DataFrame({"y_true": ["A", "B"], "y_pred": ["B", "A"]})
    mapping = {"y_true": "y_true", "y_pred": "y_pred", "_task_type": "multiclass"}

    assert common.calculate_kl_divergence(df, mapping) == pytest.approx(0.0, abs=1e-9)


def test_m6_kl_divergence_is_asymmetric():
    """KL 은 거리가 아니다 — p‖q 와 q‖p 가 다르다. 대칭 지표로 구현하면 걸린다."""
    forward = pd.DataFrame({"y_true": ["A", "A", "A", "B"], "y_pred": ["A", "A", "B", "B"]})
    backward = pd.DataFrame({"y_true": ["A", "A", "B", "B"], "y_pred": ["A", "A", "A", "B"]})
    mapping = {"y_true": "y_true", "y_pred": "y_pred", "_task_type": "multiclass"}

    assert common.calculate_kl_divergence(forward, mapping) != pytest.approx(
        common.calculate_kl_divergence(backward, mapping)
    )
