from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_car_presence_with_metrics(
    variance_dict: Dict[int, Tuple[float, int]], threshold: float
) -> Tuple[Dict[str, float], List[List[int]]]:
    """Predict presence of a car in each frame based on brightness variance
       and evaluate prediction quality, also find intervals of car presence.

    Args:
        variance_dict (Dict[int, Tuple[float, int]]): Dictionary with frame number
        as key and a tuple of variance values and original labels as value.
        threshold (float): Threshold value for determining presence of a car.

    Returns:
        Dict[str, float]: Dictionary with metrics to evaluate prediction quality.
        List[List[int]]: List of intervals with car presence in frames.
    """

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    car_intervals: List[List[int]] = []
    current_interval: List[int] = []

    for frame_number, (variance, label) in variance_dict.items():
        predicted_label = 1 if variance > threshold else 0
        predicted_labels.append(predicted_label)
        true_labels.append(label)

        # Detecting car presence intervals
        if predicted_label == 1:
            if not current_interval:
                current_interval = [frame_number, frame_number]
            else:
                current_interval[1] = frame_number
        else:
            if current_interval:
                car_intervals.append(current_interval)
                current_interval = []

    # Add the last interval if it ends with a car present
    if current_interval:
        car_intervals.append(current_interval)

    # Calculating metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics, car_intervals


def find_best_threshold(
    variance_dict: Dict[int, Tuple[float, int]],
    threshold_range: Optional[np.ndarray] = None,
) -> float:
    """Find the best threshold for car presence prediction based on F1-score.

    Args:
    variance_dict (dict): Dictionary with frame numbers as keys and
        tuples of variance values and original labels as values.
    threshold_range (np.ndarray | None): Thresholds list to evaluate.
        If None - using np.linspace(min_variance, max_variance, 100).
        Defaults to None.

    Returns:
    float: The threshold value with the highest F1-score.
    """

    # Extracting variance values and finding the min and max
    variances = [variance for variance, _ in variance_dict.values()]
    min_variance = min(variances)
    max_variance = max(variances)

    # Generating a range of threshold values
    threshold_range = np.linspace(min_variance, max_variance, num=100, dtype=float)
    best_threshold = None
    best_f1_score = 0.0

    for threshold in threshold_range:
        metrics, _ = predict_car_presence_with_metrics(variance_dict, threshold)
        f1_score = metrics["f1_score"]
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold

    return best_threshold  # type: ignore
