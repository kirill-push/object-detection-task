import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from object_detection_task.baseline.baseline import process_all_videos
from object_detection_task.data.preprocess_video import AnnotationManager


def predict_car_presence_with_metrics(
    variance_dict: Dict[Any, Tuple[float, int]],
    threshold: float,
    intervals: bool = True,
) -> Tuple[Dict[str, float], List[List[int]]]:
    """Predict presence of a car in each frame based on brightness variance
       and evaluate prediction quality, also find intervals of car presence.

    Args:
        variance_dict (Dict[Any, Tuple[float, int]]): Dictionary with frame number
        as key and a tuple of variance values and original labels as value.
        threshold (float): Threshold value for determining presence of a car.
        intervals (bool): If True, then compute intervals for video.
            Defaults to True.

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
        if intervals:
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
    if intervals and current_interval:
        car_intervals.append(current_interval)

    # Calculating metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    metrics = {
        "f1_score": f1,
        "recall": recall,
        "precision": precision,
    }
    if not intervals:
        car_intervals = []
    return metrics, car_intervals  # type: ignore


def find_best_threshold(
    variance_dict: Dict[Tuple[str, int], Tuple[float, int]],
    threshold_range: Optional[np.ndarray] = None,
) -> float:
    """Find the best threshold for car presence prediction based on F1-score.

    Args:
    variance_dict (Dict[Tuple[str, int], Tuple[float, int]]): Dictionary with
        (video_name, n_frame) as keys and tuples of variance values and original labels
        as values.
    threshold_range (np.ndarray | None): Thresholds list to evaluate.
        If None - using np.linspace(min_variance, max_variance, 1000).
        Defaults to None.

    Returns:
    float: The threshold value with the highest F1-score.
    """

    # Extracting variance values and finding the min and max
    variances = [variance for variance, _ in variance_dict.values()]
    min_variance = min(variances)
    max_variance = max(variances)

    # Generating a range of threshold values
    threshold_range = np.linspace(min_variance, max_variance, num=1000, dtype=float)
    best_threshold = None
    best_f1_score = 0.0

    for threshold in tqdm(threshold_range):
        metrics, _ = predict_car_presence_with_metrics(variance_dict, threshold, False)
        f1_score = metrics["f1_score"]
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_threshold = threshold

    return best_threshold  # type: ignore


if __name__ == "__main__":
    # Create parser and initialize arguments
    parser = argparse.ArgumentParser(description="Run baseline")
    parser.add_argument(
        "-v",
        "--video_to_val",
        nargs="+",
        default=None,
        help="One video name or list of video names to validate.",
    )
    parser.add_argument(
        "-d",
        "--path_to_video_dir",
        default="resources/videos",
        help="Path to directory with all videos.",
    )
    parser.add_argument(
        "-i",
        "--file_path_intervals",
        default="resources/time_intervals.json",
        help="Path to intervals annotation.",
    )
    parser.add_argument(
        "-p",
        "--file_path_polygons",
        default="resources/polygons.json",
        help="Path to polygons annotation.",
    )
    parser.add_argument(
        "-r",
        "--path_to_resources",
        default="resources",
        help="Path to the resources directory.",
    )

    # Collect arguments
    args = parser.parse_args()

    # Use collected arguments
    video_to_val = args.video_to_val
    path_to_video_dir = args.path_to_video_dir
    file_path_intervals = args.file_path_intervals
    file_path_polygons = args.file_path_polygons
    path_to_resources = args.path_to_resources

    intervals_manager = AnnotationManager(
        file_path_intervals,
        "intervals",
    )
    # Video 16 and video 17 are copies (same or in worse quality) of video 4 and 3
    video_to_del = ["video_16.mp4", "video_17.mp4"]
    if video_to_val is None:
        video_to_val = []
    video_list = [
        video
        for video in intervals_manager.video_list
        if video not in video_to_val and video not in video_to_del
    ]
    variance_dict_train = process_all_videos(
        path_to_video_dir,
        file_path_intervals,
        file_path_polygons,
        video_list,
    )
    best_threshold = find_best_threshold(variance_dict_train)

    if len(video_to_val) != 0:
        variance_dict_vals = process_all_videos(
            path_to_video_dir,
            file_path_intervals,
            file_path_polygons,
            video_to_val,
        )

        metrics, _ = predict_car_presence_with_metrics(
            variance_dict_vals, best_threshold, False
        )
        with open(
            os.path.join(path_to_resources, "baseline_metrics_val.json"), "w"
        ) as f:
            json.dump(metrics, f)
