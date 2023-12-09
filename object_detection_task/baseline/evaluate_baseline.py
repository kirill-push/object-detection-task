from typing import Dict, List, Tuple

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
