import os
from typing import Dict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from object_detection_task.data.preprocess_video import (
    extract_frames,
    get_labels,
    read_annotations,
)
from object_detection_task.detector.run_detector import predict


def compute_metrics(
    video_path: str,
    intervals_path: str,
    polygons_path: str,
    thresholds_path: str = "resources/thresholds.json",
) -> Dict[str, float]:
    # Process params
    video_name = os.path.basename(video_path)
    intervals = read_annotations(intervals_path)[video_name]
    # Get video length to compute labels
    video_length = len(extract_frames(video_path))
    # Get labels list from intervals
    labels = get_labels(intervals, video_length)
    # Make predictions
    predictions_dict = predict(
        video_path=video_path,
        polygons_path=polygons_path,
        thresholds_path=thresholds_path,
    )
    predictions = list(predictions_dict.values())

    # Compute metrics from predictions
    metrics_dict = {
        "recall": recall_score(labels, predictions, zero_division=1),
        "f1": f1_score(labels, predictions, zero_division=1),
        "precision": precision_score(labels, predictions, zero_division=1),
        "accuracy": accuracy_score(labels, predictions),
    }
    return metrics_dict
