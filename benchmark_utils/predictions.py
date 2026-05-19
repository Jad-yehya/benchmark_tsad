import numpy as np


def cutoff_scores(anomaly_scores, cutoff=None):
    """Turn anomaly scores into binary predictions using a contamination rate.

    Larger scores are assumed to be more anomalous. NaN entries are preserved
    as ``-1`` ignore labels so they are masked by the objective.
    """
    if cutoff is None:
        return None

    validate_cutoff(cutoff)

    scores = np.asarray(anomaly_scores)
    predictions = np.full(scores.shape, -1, dtype=int)
    valid = ~np.isnan(scores)
    if not np.any(valid):
        return predictions

    threshold = np.quantile(scores[valid], 1 - cutoff)

    predictions[valid] = (scores[valid] >= threshold).astype(int)
    return predictions


def validate_cutoff(cutoff):
    if cutoff is None:
        raise ValueError("cutoff must be provided.")
    if not 0 < cutoff < 1:
        raise ValueError(
            "cutoff must be in (0, 1), "
            f"got {cutoff!r}."
        )
