import numpy as np


def soft_precision(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   detection_range=3,
                   return_counts=False
                   ):
    """Ratio of correctly detected anomalies to the number of total anomalies
    If an anomaly is detected detection_range windows before or after
    the true anomaly, we consider it as detected
    e.g. detection_range=3, if anomaly detected at timestamp 5,
    we consider y_pred correct if there is an anomaly at timestamp 2,..., 8

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        detection_range : int, default=3
            Range in which the anomaly is considered correctly detected

    Returns
    -------
        Recall : float
            Precision score
        em : int
            Number of exact matches
        da : int
            Number of detected anomalies
        fa : int
            Number of false anomalies
    """
    # EM : Exact Match
    em = 0
    # DA : Detected Anomaly
    da = 0
    # FA : False Anomaly
    fa = 0

    # TFDIR = (EM + DA) / (EM + DA + FA)

    # Counting exact matches
    for i in range(len(y_true)):
        if y_true[i] == 1 and (y_true[i] == y_pred[i]):
            em += 1

    # False anomaly and detected anomalies
    for i in range(len(y_true)):

        left = max(0, i-detection_range)
        right = min(len(y_true), i+detection_range+1)

        if y_pred[i] == 1 and (
                y_true[left:right] == 0).all():
            fa += 1

        if y_true[i] == 1 and (
                y_pred[left:right] == 1).any():
            da += 1

    # Removing exact matches from detected anomalies because they are
    # counted twice
    da -= em

    if return_counts:
        if em + da + fa == 0:
            return 0, em, da, fa

        return (em + da) / (em + da + fa), em, da, fa

    if em + da + fa == 0:
        return 0
    return (em + da) / (em + da + fa)


def soft_recall(y_true: np.ndarray,
                y_pred: np.ndarray,
                detection_range=3,
                return_counts=False,
                ):
    """
    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        detection_range : int, default=3
            Range in which the anomaly is considered correctly detected

    Returns
    -------
        recall : float
            recall score
        em : int
            Number of exact matches
        da : int
            Number of detected anomalies
        ma : int
            Number of missed anomalies
    """
    # EM : Exact Match
    em = 0
    # DA : Detected Anomaly
    da = 0
    # MA : Missed Anomaly
    ma = 0
    # DAIR = (EM + DA) / (EM + DA + MA)

    # Counting exact matches
    for i in range(len(y_true)):
        if y_true[i] == 1 and (y_true[i] == y_pred[i]):
            em += 1

    # Missing values and detected anomalies
    for i in range(len(y_true)):

        left = max(0, i-detection_range)
        right = min(len(y_true), i+detection_range+1)

        if y_true[i] == 1 and (
                y_pred[left:right] == 0).all():
            ma += 1

        if y_true[i] == 1 and (
                y_pred[left:right] == 1).any():
            da += 1

    # Removing exact matches from detected anomalies because they are
    # counted twice
    da -= em

    if return_counts:
        if em + da + ma == 0:
            return 0, em, da, ma

        return (em + da) / (em + da + ma), em, da, ma

    if em + da + ma == 0:
        return 0
    return (em + da) / (em + da + ma)


def ctt(y_true: np.ndarray, y_pred: np.ndarray, return_signed: bool = False):
    """
    Candidate To Target: Means distance between predicted anomaly and
    its closest true anomaly.

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        return_signed : bool, default=False
            If True, return the signed distance.
            If False, return the absolute distance.

    Returns
    -------
        ctt : float
            Candidate To Target time
    """
    if np.sum(y_true) == 0:
        # No anomalies to detect
        return float('inf')
    elif np.sum(y_pred) == 0:
        # No anomalies detected
        return 0

    tot_dist = 0

    # Indices of true anomalies
    true_anomalies = np.where(y_true == 1)[0]

    for i in np.where(y_pred == 1)[0]:
        if len(true_anomalies) > 0:
            # signed distances to all true anomalies
            signed_dists = true_anomalies - i
            if return_signed:
                # signed distance with the smallest absolute value
                min_dist = signed_dists[np.argmin(np.abs(signed_dists))]
            else:
                # absolute distance with the smallest value
                min_dist = np.abs(signed_dists).min()
            tot_dist += min_dist

    return tot_dist / np.sum(y_pred)


def ttc(y_true: np.ndarray, y_pred: np.ndarray, return_signed: bool = False):
    """
    Target To Candidate: Means distance between true anomaly and
    its closest predicted anomaly.

    Parameters
    ----------
        y_true : np.ndarray
            Ground truth labels
        y_pred : np.ndarray
            Predicted labels
        return_signed : bool, default=False
            If True, return the signed distance.
            If False, return the absolute distance.

    Returns
    -------
        ttc : float
            Target To Candidate
    """
    if np.sum(y_pred) == 0:
        # No anomalies detected
        return float('inf')
    elif np.sum(y_true) == 0:
        # No true anomalies
        return 0

    tot_dist = 0

    # Indices of predicted anomalies
    pred_anomalies = np.where(y_pred == 1)[0]

    for i in np.where(y_true == 1)[0]:
        if len(pred_anomalies) > 0:
            # signed distances to all predicted anomalies
            signed_dists = pred_anomalies - i
            if return_signed:
                # signed distance with the smallest absolute value
                min_dist = signed_dists[np.argmin(np.abs(signed_dists))]
            else:
                # absolute distance with the smallest value
                min_dist = np.abs(signed_dists).min()
            tot_dist += min_dist

    return tot_dist / np.sum(y_true)


def soft_f1(precision, recall):
    """
    Calculate the F1 score from precision and recall.

    Parameters
    ----------
        precision : float
            Precision score
        recall : float
            Recall score

    Returns
    -------
        f1 : float
            F1 score
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


# Implementation of the range metrics proposed by Tatbul et al.
# https://arxiv.org/abs/1803.03639

def extract_anomaly_ranges(labels: list[int]):
    """
    Extracts ranges of anomalies from a series of labels.

    Parameters
    ----------
        labels : List[int]
                Series of labels where 1 indicates an
                anomaly and 0 indicates normal.

    Returns
    -------
        ranges : List[Tuple[int, int]]
                Each tuple represents a range (start_index, end_index)
                where anomalies are present.
    """
    ranges = []
    start = None

    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i  # Start of a new anomaly range
        elif label == 0 and start is not None:
            ranges.append((start, i - 1))  # End of the current anomaly range
            start = None

    # Handle the case where the series ends with an anomaly
    if start is not None:
        ranges.append((start, len(labels) - 1))

    return ranges


def existence_reward(real_range, predicted_ranges):
    """
    Calculates the existence reward for a real anomaly range.

    Parameters
    ----------
        real_range : Tuple[int, int]
                A tuple representing the range of a real anomaly
                (start_index, end_index).

        predicted_ranges : List[Tuple[int, int]]
                A list of tuples representing the ranges of
                predicted anomalies.

    Returns
    -------
        reward : int
                1 if there is any overlap between the real anomaly
                range and a predicted range, 0 otherwise.
    """
    real_start, real_end = real_range

    for pred_start, pred_end in predicted_ranges:
        # Check if there is any overlap between real_range and this pred range
        if max(real_start, pred_start) <= min(real_end, pred_end):
            return 1  # Overlap exists

    return 0  # No overlap exists


def cardinality_factor(real_range, predicted_ranges):
    """
    Calculates the cardinality factor for the real anomaly range.

    Parameters
    ----------
        real_range : Tuple[int, int]
                A tuple representing the real anomaly range
                as (start_index, end_index).
        predicted_ranges : List[Tuple[int, int]]
                A list of tuples representing predicted anomaly ranges,
    Returns
    -------
        cardinality : float
                The cardinality factor for the real anomaly range.
    """
    overlaps = 0
    real_start, real_end = real_range

    for pred_start, pred_end in predicted_ranges:
        if max(real_start, pred_start) <= min(real_end, pred_end):
            overlaps += 1

    if overlaps <= 1:
        return 1.0
    else:
        return 1.0 / overlaps


def positional_bias(i, anomaly_length, bias_type='flat'):
    """
    Defines the positional bias function δ(i, AnomalyLength).

    Parameters
    ----------
        i : int
            The current position in the anomaly range.
        anomaly_length : int
            The total length of the real anomaly range.

        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns
    -------
        float: The bias value for the current position.
    """
    if bias_type == 'flat':
        return 1.0
    elif bias_type == 'front':
        return anomaly_length - i + 1
    elif bias_type == 'back':
        return i
    elif bias_type == 'middle':
        if i <= anomaly_length // 2:
            return i
        else:
            return anomaly_length - i + 1
    return 1.0


def overlap_size(real_range, overlap_set, anomaly_length, bias_type='flat'):
    """
    Calculates the overlap size weighted by positional bias.

    Parameters
    ----------
        real_range : Tuple[int, int]
            A tuple representing the real anomaly range as
            (start_index, end_index).
        overlap_set : Set[int]
            A set of indices where the real anomaly range overlaps with
            predicted ranges.
        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns:
        float: The weighted overlap size based on positional bias.
    """
    overlap_value = 0
    max_value = 0

    for i in range(real_range[0], real_range[1] + 1):
        bias = positional_bias(i - real_range[0] + 1,
                               anomaly_length, bias_type)
        max_value += bias
        if i in overlap_set:
            overlap_value += bias

    return overlap_value / max_value


def overlap_reward(real_range, predicted_ranges, bias_type='flat'):
    """
    Calculates the overlap reward for a real anomaly range.
    Parameters
    ----------
        real_range : Tuple[int, int]
            A tuple representing the real anomaly range as
            (start_index, end_index).
        predicted_ranges : List[Tuple[int, int]]
            A list of tuples representing predicted anomaly ranges.
        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns
    -------
        float: The overlap reward for the real anomaly range.
    """
    real_start, real_end = real_range
    anomaly_length = real_end - real_start + 1
    total_overlap = set()

    for pred_start, pred_end in predicted_ranges:
        overlap = set(range(max(real_start, pred_start),
                            min(real_end, pred_end) + 1))
        total_overlap.update(overlap)

    if not total_overlap:
        return 0.0

    cardinality = cardinality_factor(real_range, predicted_ranges)
    size = overlap_size(real_range, total_overlap, anomaly_length, bias_type)

    return cardinality * size


def recall_t(real_ranges, predicted_ranges, alpha=0.5, bias_type='flat'):
    """
    Calculates the range-based Recall as defined in the paper.

    Parameters
    ----------
        real_ranges : List[Tuple[int, int]]
            List of real anomaly ranges, where each range is a
            tuple (start_index, end_index).
        predicted_ranges : List[Tuple[int, int]]
            List of predicted anomaly ranges, where each range is a
            tuple (start_index, end_index).
        alpha : float
            The weight to assign to the existence reward.
        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns
    -------
        float: The range-based recall score.
    """
    recall_sum = 0

    for real_range in real_ranges:
        existence = existence_reward(real_range, predicted_ranges)
        overlap = overlap_reward(real_range, predicted_ranges,
                                 bias_type=bias_type)
        recall_sum += alpha * existence + (1 - alpha) * overlap

    return recall_sum / len(real_ranges) if real_ranges else 0


def precision_t(real_ranges, predicted_ranges, bias_type='flat'):
    """
    Calculates the range-based Precision as defined in the paper.

    Parameters
    ----------
        real_ranges : List[Tuple[int, int]]
            List of real anomaly ranges, where each range is a
            tuple (start_index, end_index).
        predicted_ranges : List[Tuple[int, int]]
            List of predicted anomaly ranges, where each range is a
            tuple (start_index, end_index).
        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns
    -------
        float: The range-based precision score.
    """
    precision_sum = 0

    for predicted_range in predicted_ranges:
        overlap = overlap_reward(predicted_range, real_ranges,
                                 bias_type=bias_type)
        precision_sum += overlap

    return precision_sum / len(predicted_ranges) if predicted_ranges else 0


def f1_t(real_ranges, predicted_ranges, alpha=0.5, bias_type='flat'):
    """
    Calculates the range-based F1 score as defined in the paper.

    Parameters
    ----------
        real_ranges : List[Tuple[int, int]]
            List of real anomaly ranges, where each range is a
            tuple (start_index, end_index).
        predicted_ranges : List[Tuple[int, int]]
            List of predicted anomaly ranges, where each range is a
            tuple (start_index, end_index).
        alpha : float
            The weight to assign to the existence reward.
        bias_type : str
            The type of positional bias ('flat', 'front', 'back', 'middle').

    Returns
    -------
        float: The range-based F1 score.
    """
    recall_score = recall_t(real_ranges, predicted_ranges, alpha, bias_type)
    precision_score = precision_t(real_ranges, predicted_ranges, bias_type)

    if recall_score + precision_score == 0:
        return 0

    return 2 * (recall_score*precision_score)/(recall_score+precision_score)
