import pytest  # noqa

import numpy as np
from benchmark_utils.metrics import (
    soft_precision, soft_recall, soft_f1, ctt, ttc,
    extract_anomaly_ranges, existence_reward, cardinality_factor,
    positional_bias, overlap_size, overlap_reward,
    recall_t, precision_t, f1_t
)


def test_soft_precision():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert soft_precision(y1, y1, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p1 = y1.copy()
    assert soft_precision(y1, p1, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p2 = y1.copy()
    p2[7] = 0
    assert soft_precision(y1, p2, detection_range=1,
                          return_counts=True) == (1.0, 1, 0, 0)

    p3 = y1.copy()
    p3[8] = 1
    assert soft_precision(y1, p3, detection_range=1,
                          return_counts=True) == (1.0, 2, 0, 0)

    p4 = y1.copy()
    p4[9] = 1
    assert soft_precision(y1, p4, detection_range=1,
                          return_counts=True) == (2/3, 2, 0, 1)

    p5 = np.zeros(10)
    p5[0] = 1
    assert soft_precision(y1, p5, detection_range=1,
                          return_counts=True) == (0.0, 0, 0, 1)

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert soft_precision(y1, p6, detection_range=1,
                          return_counts=True) == (1.0, 0, 2, 0)


def test_soft_recall():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert soft_recall(y1, y1, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p1 = y1.copy()
    assert soft_recall(y1, p1, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p2 = y1.copy()
    p2[7] = 0
    assert soft_recall(y1, p2, detection_range=1,
                       return_counts=True) == (0.5, 1, 0, 1)

    p3 = y1.copy()
    p3[8] = 1
    assert soft_recall(y1, p3, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p4 = y1.copy()
    p4[9] = 1
    assert soft_recall(y1, p4, detection_range=1,
                       return_counts=True) == (1.0, 2, 0, 0)

    p5 = np.zeros(10)
    p5[0] = 1
    assert soft_recall(y1, p5, detection_range=1,
                       return_counts=True) == (0.0, 0, 0, 2)

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert soft_recall(y1, p6, detection_range=1,
                       return_counts=True) == (1.0, 0, 2, 0)


def test_soft_f1():

    y1 = np.zeros(10)
    y1[3] = y1[7] = 1

    precision = soft_precision(y1, y1, detection_range=1)
    recall = soft_recall(y1, y1, detection_range=1)

    assert soft_f1(y1, y1, detection_range=1) == 1.0

    p2 = y1.copy()
    p2[7] = 0
    precision = soft_precision(y1, p2, detection_range=1)
    recall = soft_recall(y1, p2, detection_range=1)

    assert soft_f1(y1, p2, detection_range=1) == 2 * \
        precision * recall / (precision + recall)

    p3 = y1.copy()
    p3[8] = 1

    precision = soft_precision(y1, p3, detection_range=1)
    recall = soft_recall(y1, p3, detection_range=1)

    assert soft_f1(y1, p3, detection_range=1) == 2 * \
        precision * recall / (precision + recall)


def test_ctt():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1

    p1 = y1.copy()
    assert ctt(y1, p1, return_signed=True) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ctt(y1, p2, return_signed=True) == 0.0

    p3 = y1.copy()
    p3[8] = 1
    assert ctt(y1, p3, return_signed=True) == -1/3

    p4 = y1.copy()
    p4[9] = 1
    assert ctt(y1, p4, return_signed=True) == -2/3

    p5 = np.zeros(10)
    p5[0] = 1
    assert ctt(y1, p5, return_signed=True) == 3.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ctt(y1, p6, return_signed=True) == -1.0

    y2 = np.zeros(10)
    p7 = np.zeros(10)
    p7[4] = p7[8] = 1
    assert ctt(y2, p7, return_signed=True) == float('inf')

    p8 = np.zeros(10)
    assert ctt(y2, p8, return_signed=True) == float('inf')

    y3 = np.zeros(10)
    y3[2] = 1
    p9 = np.zeros(10)
    assert ctt(y3, p9, return_signed=True) == 0.0


def test_ttc():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert ttc(y1, y1, return_signed=True) == 0.0

    p1 = y1.copy()
    assert ttc(y1, p1, return_signed=True) == 0.0

    p2 = y1.copy()
    p2[7] = 0
    assert ttc(y1, p2, return_signed=True) == -2.0

    p3 = y1.copy()
    p3[8] = 1
    assert ttc(y1, p3, return_signed=True) == 0.0

    p4 = y1.copy()
    p4[9] = 1
    assert ttc(y1, p4, return_signed=True) == 0.0

    p5 = np.zeros(10)
    p5[0] = 1
    assert ttc(y1, p5, return_signed=True) == -5.0

    p6 = np.zeros(10)
    p6[4] = p6[8] = 1
    assert ttc(y1, p6, return_signed=True) == 1.0

    y2 = np.zeros(10)
    p7 = np.zeros(10)
    p7[4] = p7[8] = 1
    assert ttc(y2, p7, return_signed=True) == 0

    p8 = np.zeros(10)
    assert ttc(y2, p8, return_signed=True) == float('inf')

    y3 = np.zeros(10)
    y3[2] = 1
    p9 = np.zeros(10)
    assert ttc(y3, p9, return_signed=True) == float('inf')


def test_extract_anomaly_ranges():
    y1 = np.zeros(10)
    y1[3] = y1[7] = 1
    assert extract_anomaly_ranges(y1) == [(3, 3), (7, 7)]

    y2 = np.zeros(10)
    y2[3] = y2[4] = y2[5] = 1
    assert extract_anomaly_ranges(y2) == [(3, 5)]

    y3 = np.ones(10)
    assert extract_anomaly_ranges(y3) == [(0, 9)]

    y4 = np.zeros(10)
    assert extract_anomaly_ranges(y4) == []

    y5 = np.zeros(10)
    y5[0] = y5[1] = y5[2] = 1
    y5[4] = y5[5] = y5[6] = 1
    y5[8] = y5[9] = 1
    assert extract_anomaly_ranges(y5) == [(0, 2), (4, 6), (8, 9)]


def test_existence_reward():
    real_range = (5, 10)
    predicted_ranges = [(8, 12), (15, 20)]
    assert existence_reward(real_range, predicted_ranges) == 1

    real_range = (5, 10)
    predicted_ranges = [(11, 15), (20, 25)]
    assert existence_reward(real_range, predicted_ranges) == 0

    real_range = (5, 10)
    predicted_ranges = [(5, 10), (20, 25)]
    assert existence_reward(real_range, predicted_ranges) == 1

    real_range = (8, 9)
    predicted_ranges = [(5, 10), (20, 25)]
    assert existence_reward(real_range, predicted_ranges) == 1

    real_range = (5, 10)
    predicted_ranges = [(6, 7), (20, 25)]
    assert existence_reward(real_range, predicted_ranges) == 1

    real_range = (5, 10)
    predicted_ranges = [(11, 15), (3, 4), (6, 7)]
    assert existence_reward(real_range, predicted_ranges) == 1

    real_range = (5, 10)
    predicted_ranges = []
    assert existence_reward(real_range, predicted_ranges) == 0

    real_range = (5, 10)
    predicted_ranges = [(10, 15)]
    assert existence_reward(real_range, predicted_ranges) == 1


def test_cardinality_factor():
    real_range = (5, 10)
    predicted_ranges = [(11, 15), (16, 20)]
    assert cardinality_factor(real_range, predicted_ranges) == 1.0

    real_range = (5, 10)
    predicted_ranges = [(8, 12), (15, 20)]
    assert cardinality_factor(real_range, predicted_ranges) == 1.0

    real_range = (5, 10)
    predicted_ranges = [(8, 12), (3, 6)]
    assert cardinality_factor(real_range, predicted_ranges) == 0.5

    real_range = (5, 10)
    predicted_ranges = [(4, 6), (8, 9), (10, 15)]
    assert cardinality_factor(real_range, predicted_ranges) == 1.0 / 3

    real_range = (5, 10)
    predicted_ranges = [(5, 10)]
    assert cardinality_factor(real_range, predicted_ranges) == 1.0

    real_range = (5, 15)
    predicted_ranges = [(5, 7), (8, 10), (12, 14)]
    assert cardinality_factor(real_range, predicted_ranges) == 1.0 / 3

    real_range = (5, 10)
    predicted_ranges = []
    assert cardinality_factor(real_range, predicted_ranges) == 1.0


def test_positional_bias():
    # Flat bias
    assert positional_bias(1, 10, 'flat') == 1.0
    assert positional_bias(5, 10, 'flat') == 1.0
    assert positional_bias(10, 10, 'flat') == 1.0

    # Front bias
    assert positional_bias(1, 10, 'front') == 10
    assert positional_bias(5, 10, 'front') == 6
    assert positional_bias(10, 10, 'front') == 1

    # Back bias
    assert positional_bias(1, 10, 'back') == 1
    assert positional_bias(5, 10, 'back') == 5
    assert positional_bias(10, 10, 'back') == 10

    # Middle bias
    assert positional_bias(1, 10, 'middle') == 1
    assert positional_bias(5, 10, 'middle') == 5
    assert positional_bias(10, 10, 'middle') == 1

    assert positional_bias(1, 9, 'middle') == 1
    assert positional_bias(5, 9, 'middle') == 5
    assert positional_bias(9, 9, 'middle') == 1

    # Default case (fallback)
    assert positional_bias(1, 10) == 1.0
    assert positional_bias(5, 10) == 1.0
    assert positional_bias(10, 10) == 1.0


def test_overlap_size():
    real_range = (1, 5)
    overlap_set = {1, 2, 3, 4, 5}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set, anomaly_length, 'flat') == 1.0

    real_range = (1, 5)
    overlap_set = {1, 2, 3, 4, 5}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set,
                        anomaly_length, 'front') == 1.0

    real_range = (1, 5)
    overlap_set = {1, 2, 3, 4, 5}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set, anomaly_length, 'back') == 1.0

    real_range = (1, 5)
    overlap_set = {1, 2, 3, 4, 5}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set,
                        anomaly_length, 'middle') == 1.0

    real_range = (1, 5)
    overlap_set = {3, 4, 5}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set, anomaly_length, 'back') == (
        3 + 4 + 5) / (1 + 2 + 3 + 4 + 5)

    real_range = (1, 5)
    overlap_set = {2, 3, 4}
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set, anomaly_length, 'middle') == (
        2 + 3 + 2) / (1 + 2 + 3 + 2 + 1)

    real_range = (1, 5)
    overlap_set = set()
    anomaly_length = 5
    assert overlap_size(real_range, overlap_set, anomaly_length, 'flat') == 0.0


def test_overlap_reward():
    real_range = (10, 20)
    predicted_ranges = [(0, 5), (25, 30)]
    assert overlap_reward(real_range, predicted_ranges) == 0.0

    real_range = (10, 20)
    predicted_ranges = [(10, 20)]
    assert overlap_reward(real_range, predicted_ranges) == 1.0

    real_range = (10, 20)
    predicted_ranges = [(15, 25)]
    assert overlap_reward(real_range, predicted_ranges) > 0.0
    assert overlap_reward(real_range, predicted_ranges) < 1.0

    real_range = (10, 20)
    predicted_ranges = [(5, 15), (18, 25)]
    assert overlap_reward(real_range, predicted_ranges) > 0.0

    real_range = (10, 20)
    predicted_ranges = []
    assert overlap_reward(real_range, predicted_ranges) == 0.0

    real_range = (10, 20)
    predicted_ranges = [(15, 25)]
    for bias_type in ['flat', 'front', 'back', 'middle']:
        reward = overlap_reward(
            real_range, predicted_ranges, bias_type=bias_type)
        assert reward >= 0.0  # Ensure reward is non-negative

    real_range = (10, 20)
    predicted_ranges = [(10, 15), (16, 20)]
    assert overlap_reward(
        real_range, predicted_ranges) < overlap_reward(real_range, [(10, 20)])
    # Test multiple ranges covering the real range result in a lower reward
    # due to the cardinality factor.

    real_range = (10, 10)
    predicted_ranges = [(10, 10)]
    assert overlap_reward(real_range, predicted_ranges) == 1.0

    real_range = (10, 10)
    predicted_ranges = [(9, 11)]
    assert overlap_reward(real_range, predicted_ranges) == 1.0


def test_recall_t():
    real_ranges = [(10, 20)]
    predicted_ranges = [(10, 20)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    assert abs(result - 1.0) < 1e-6

    real_ranges = [(10, 20)]
    predicted_ranges = [(15, 25)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    assert 0 < result < 1.0

    real_ranges = [(10, 20)]
    predicted_ranges = [(21, 30)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    assert abs(result - 0.0) < 1e-6

    real_ranges = [(10, 20), (30, 40)]
    predicted_ranges = [(15, 25), (35, 45)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    assert 0 < result < 1.0

    real_ranges = [(10, 20)]
    predicted_ranges = [(21, 30)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=1.0, bias_type='flat')
    assert abs(result - 0.0) < 1e-6

    predicted_ranges = [(15, 25)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=1.0, bias_type='flat')
    assert abs(result - 1.0) < 1e-6

    real_ranges = [(10, 20)]
    predicted_ranges = [(15, 25)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.0, bias_type='flat')
    assert 0 < result < 1.0

    real_ranges = [(10, 20)]
    predicted_ranges = [(10, 15)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='front')
    assert result > recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')

    real_ranges = [(10, 20)]
    predicted_ranges = [(15, 20)]
    result = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='back')
    assert result > recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')


def test_precision_t():
    real_ranges = [(10, 20), (30, 40), (50, 60)]
    predicted_ranges = [(15, 25), (35, 45), (55, 65)]

    assert precision_t(real_ranges, predicted_ranges, bias_type='front') > 0

    assert precision_t(real_ranges, predicted_ranges, bias_type='back') > 0

    assert precision_t(real_ranges, predicted_ranges, bias_type='middle') > 0

    no_overlap_predicted_ranges = [(70, 80)]
    assert precision_t(
        real_ranges, no_overlap_predicted_ranges, bias_type='flat') == 0.0

    partial_overlap_predicted_ranges = [(15, 25), (70, 80)]
    assert precision_t(
        real_ranges, partial_overlap_predicted_ranges, bias_type='flat') > 0

    different_anomaly_length_predicted_ranges = [(10, 15), (32, 38), (52, 57)]
    assert precision_t(
        real_ranges,
        different_anomaly_length_predicted_ranges,
        bias_type='flat') > 0

    empty_predicted_ranges = []
    assert precision_t(
        real_ranges,
        empty_predicted_ranges,
        bias_type='flat') == 0.0

    empty_real_ranges = []
    assert precision_t(
        empty_real_ranges,
        predicted_ranges,
        bias_type='flat') == 0.0


def test_f1_t():
    real_ranges = [(10, 20), (30, 40), (50, 60)]
    predicted_ranges = [(15, 25), (35, 45), (55, 65)]

    recall_score = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    precision_score = precision_t(
        real_ranges, predicted_ranges, bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        predicted_ranges,
        alpha=0.5,
        bias_type='flat') == expected_f1

    recall_score = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='front')
    precision_score = precision_t(
        real_ranges, predicted_ranges, bias_type='front')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        predicted_ranges,
        alpha=0.5,
        bias_type='front') == expected_f1

    recall_score = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='back')
    precision_score = precision_t(
        real_ranges, predicted_ranges, bias_type='back')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        predicted_ranges,
        alpha=0.5,
        bias_type='back') == expected_f1

    recall_score = recall_t(
        real_ranges, predicted_ranges, alpha=0.5, bias_type='middle')
    precision_score = precision_t(
        real_ranges, predicted_ranges, bias_type='middle')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        predicted_ranges,
        alpha=0.5,
        bias_type='middle') == expected_f1

    no_overlap_predicted_ranges = [(70, 80)]
    recall_score = recall_t(
        real_ranges, no_overlap_predicted_ranges, alpha=0.5, bias_type='flat')
    precision_score = precision_t(
        real_ranges,
        no_overlap_predicted_ranges,
        bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        no_overlap_predicted_ranges,
        alpha=0.5,
        bias_type='flat') == expected_f1

    partial_overlap_predicted_ranges = [(15, 25), (70, 80)]
    recall_score = recall_t(
        real_ranges,
        partial_overlap_predicted_ranges,
        alpha=0.5,
        bias_type='flat')
    precision_score = precision_t(
        real_ranges,
        partial_overlap_predicted_ranges,
        bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(
        real_ranges,
        partial_overlap_predicted_ranges,
        alpha=0.5,
        bias_type='flat') == expected_f1

    different_anomaly_length_predicted_ranges = [(10, 15), (32, 38), (52, 57)]
    recall_score = recall_t(
        real_ranges,
        different_anomaly_length_predicted_ranges,
        alpha=0.5, bias_type='flat')
    precision_score = precision_t(
        real_ranges,
        different_anomaly_length_predicted_ranges,
        bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(real_ranges, different_anomaly_length_predicted_ranges,
                alpha=0.5, bias_type='flat') == expected_f1

    # Test case 8: Empty predicted ranges
    empty_predicted_ranges = []
    recall_score = recall_t(
        real_ranges, empty_predicted_ranges, alpha=0.5, bias_type='flat')
    precision_score = precision_t(
        real_ranges, empty_predicted_ranges, bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(real_ranges, empty_predicted_ranges,
                alpha=0.5, bias_type='flat') == expected_f1

    # Test case 9: Empty real ranges
    empty_real_ranges = []
    recall_score = recall_t(
        empty_real_ranges, predicted_ranges, alpha=0.5, bias_type='flat')
    precision_score = precision_t(
        empty_real_ranges, predicted_ranges, bias_type='flat')
    expected_f1 = 2 * (
        recall_score * precision_score
    ) / (
        recall_score + precision_score
    ) if (recall_score + precision_score) != 0 else 0
    assert f1_t(empty_real_ranges, predicted_ranges,
                alpha=0.5, bias_type='flat') == expected_f1
