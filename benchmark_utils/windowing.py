import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_windows(X, window_size=32, stride=1, padding=False):
    """Create a windowed view of the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features, n_times).
    window_size : int
        Size of the sliding window.
    stride : int
        Stride of the sliding window.

    Returns
    -------
    windows : np.ndarray
        A windowed view of the data in shape:
        (n_eff_samples, window_size, n_features)
    """

    if padding:
        n_samples, n_features, n_times = X.shape
        n_pad = (window_size - stride + n_times % stride) % stride
        pad_width = ((0, 0), (0, 0), (0, n_pad))
        X = np.pad(X, pad_width=pad_width, mode='constant')

    return np.lib.stride_tricks.sliding_window_view(
        X, window_shape=window_size, axis=-1
    )[..., ::stride, :].transpose(0, 2, 1, 3).reshape(
        -1, X.shape[1], window_size
    ).transpose(0, 2, 1)


def make_windowed_dataset(X, y=None, window_size=32, stride=1):
    """
    Create a DataLoader with windowed views of the data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features, n_times).
    y : np.ndarray, optional
        Target data of shape (n_samples, n_times).
    window_size : int
        Size of the sliding window.
    stride : int
        Stride of the sliding window.

    Returns
    -------
    Dataset
        A PyTorch Dataset with windowed data in shape:
        (n_eff_samples, window_size, n_features)
    """

    if window_size is not None:
        X = make_windows(X, window_size, stride)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    if y is not None:
        if window_size is not None:
            y = np.lib.stride_tricks.sliding_window_view(
                y, window_shape=window_size, axis=-1
            )[..., ::stride, :].reshape(-1, window_size)

        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor)

    return dataset


def reconstruct_from_windows(windows, stride, batch, n_features):
    """Reconstruct the original signal from overlapping windows

    Parameters
    ----------
    windows : np.ndarray
        The overlapping windows of shape (batch*n_windows, window_size, n_features)
    stride : int
        The stride used to create the windows
    batch : int
        The batch size used when creating the windows
    n_features : int
        The number of features in the original signal
    """
    # windows: (batch*n_windows, window_size, n_features)
    w = windows.shape[1]
    windows = windows.reshape(batch, -1, w, n_features)
    b, nw, ws, nf = windows.shape
    nt = (nw - 1) * stride + ws

    # allocate accumulator + counts for correct overlap averaging
    acc = np.zeros((b, nf, nt))
    cnt = np.zeros((nt,), dtype=int)

    # build index map for overlap positions
    idx = np.arange(ws)[:, None] + stride * np.arange(nw)

    # add windows efficiently
    np.add.at(acc, (slice(None), slice(None), idx.ravel()),
              windows.transpose(0, 3, 1, 2).reshape(b, nf, -1))

    # count contributions
    np.add.at(cnt, idx.ravel(), 1)

    return acc / cnt