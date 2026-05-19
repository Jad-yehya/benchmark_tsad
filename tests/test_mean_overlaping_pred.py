import numpy as np

from benchmark_utils import mean_overlaping_pred


def test_length_horizon_one_stride_one():
    # 5 windows, horizon=1, stride=1 → reconstructed signal length is 5
    preds = np.arange(5).reshape(5, 1, 1).astype(float)
    out = mean_overlaping_pred(preds, stride=1)
    assert out.shape == (5, 1)
    assert np.allclose(out.ravel(), np.arange(5))


def test_length_horizon_gt_one():
    # 4 windows, H=3, stride=1 → (4-1)*1 + 3 = 6 positions
    preds = np.ones((4, 3, 2))
    out = mean_overlaping_pred(preds, stride=1)
    assert out.shape == (6, 2)
    # every position covered, averaged value is 1.0
    assert np.allclose(out, 1.0)


def test_overlap_averages_correctly():
    # H=2, stride=1, 3 windows. Index 1 is covered by windows 0 and 1,
    # index 2 by windows 1 and 2.
    preds = np.array(
        [[[1.0], [2.0]],
         [[3.0], [4.0]],
         [[5.0], [6.0]]]
    )
    out = mean_overlaping_pred(preds, stride=1)
    # positions: 0 -> 1, 1 -> mean(2, 3) = 2.5, 2 -> mean(4, 5) = 4.5, 3 -> 6
    assert out.shape == (4, 1)
    assert np.allclose(out.ravel(), [1.0, 2.5, 4.5, 6.0])


def test_stride_gt_one_no_overlap():
    # H=2, stride=2 → windows tile end-to-end
    preds = np.array(
        [[[1.0], [2.0]],
         [[3.0], [4.0]],
         [[5.0], [6.0]]]
    )
    out = mean_overlaping_pred(preds, stride=2)
    # (3-1)*2 + 2 = 6 positions, no overlap
    assert out.shape == (6, 1)
    assert np.allclose(out.ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
