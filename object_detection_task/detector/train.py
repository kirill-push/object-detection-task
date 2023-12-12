import argparse
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def filter_vehicles(
    detections: Dict[str, Dict[str, Dict[str, List[Dict]]]],
    vehicle_class_ids: List[int],
) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
    """Filters the provided detections data to include only objects that are classified
        as vehicles.

    Parameters:
        detections (Dict[str, Dict[str, Dict[str, List[Dict]]]]): A dictionary where the
            key is the video name, and the value is another dictionary with frame
            numbers as keys and detection data as values.
            Each detection data is a dictionary containing label and detector results.
        vehicle_class_ids (List[int]): A list of class IDs that are considered as
            vehicles.

    Returns:
        Dict[str, Dict[str, Dict[str, List[Dict]]]]: Filtered version of the detections
            dictionary, containing only detections of objects classified as vehicles.
    """

    filtered_detections = {}
    for video_name, frames in detections.items():
        filtered_frames = {}
        for frame_number, frame_data in frames.items():
            # Include only detections that are classified as vehicles
            vehicle_detections = [
                detection
                for detection in frame_data["detector_result"]
                if detection["class_id"] in vehicle_class_ids
            ]
            # Update the frame data with filtered detections
            filtered_frames[frame_number] = {
                "label": frame_data["label"],
                "detector_result": vehicle_detections,
            }
        # Update the detections with filtered frames for each video
        filtered_detections[video_name] = filtered_frames
    return filtered_detections


def predict_vehicle_presence_sorted(
    frame_data: Dict[str, List[Dict[str, Union[float, int]]]],
    intersection_threshold: float = 0.25,
    confidence_threshold: float = 0.3,
) -> int:
    """Predicts the presence of a vehicle in a frame based on the detection results,
        first sorting detections by confidence level and then checking for intersection
        proportion.

    Args:
        frame_data (Dict[str, List[Dict[str, Union[float, int]]]]): Data for a single
            frame, containing 'label' and 'detector_result'.
        intersection_threshold (float, optional): Minimum proportion of intersection to
            consider a detection significant. Defaults to 0.25
        confidence_threshold (float, optional): Minimum confidence level to consider a
            detection reliable. Defaults to 0.3

    Returns:
        int: 1 if a vehicle is predicted to be present in the frame, 0 otherwise.
    """
    if not frame_data["detector_result"]:
        return 0  # No objects detected, so no vehicle present

    # Sort detections by confidence level in descending order
    sorted_detections = sorted(
        frame_data["detector_result"], key=lambda x: x["confidence"], reverse=True
    )

    # Check each sorted detection for significant intersection and high confidence
    for detection in sorted_detections:
        if (
            detection["proportion_of_intersection"] >= intersection_threshold
            and detection["confidence"] >= confidence_threshold
        ):
            return 1  # Vehicle detected

    # No vehicle detected after considering confidence and intersection thresholds
    return 0


def predict_vehicle_in_video(
    video_data: Dict[str, Dict[str, List[Dict]]],
    intersection_threshold: float = 0.25,
    confidence_threshold: float = 0.3,
) -> Dict[str, int]:
    """Predicts the presence of a vehicle for each frame in a video.

    Args:
        video_data (Dict[str, Dict[str, List[Dict]]]): The data for a single video,
            containing frame data.
        intersection_threshold (float, optional): Minimum proportion of intersection to
            consider a detection significant. Defaults to 0.25
        confidence_threshold (float, optional): Minimum confidence level to consider a
            detection reliable. Defaults to 0.3

    Returns:
        Dict[str, int]: A dictionary with frame numbers as keys and vehicle presence
            predictions (1/0) as values.
    """
    predictions = {}
    for frame_number, frame_data in video_data.items():
        predictions[str(frame_number)] = predict_vehicle_presence_sorted(
            frame_data, intersection_threshold, confidence_threshold
        )
    return predictions


def calculate_metrics(
    actual_labels: List[int], predicted_labels: List[int]
) -> Dict[str, float]:
    """Calculate F1-score, Recall, Accuracy, Precision, based on actual and predicted
        labels.

    Args:
        actual_labels (List[int]): The actual labels for each frame in the video.
        predicted_labels (List[int]): The predicted labels for each frame in the video.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
    """
    y_true = np.array(actual_labels)
    y_pred = np.array(predicted_labels)

    metrics = {
        "f1_score": f1_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=1),
    }
    return metrics


def calculate_global_metrics(
    filtered_detections_data: Dict[str, Dict[str, Dict[str, List[Dict]]]],
    num_intersection_thresholds: int = 50,
    num_confidence_thresholds: int = 50,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Calculate global metrics by aggregating all predictions and labels across videos
        for different thresholds.

    Args:
        filtered_detections_data (Dict[str, Dict[str, Dict[str, List[Dict]]]]): Filtered
            detections data for all videos.
        num_intersection_thresholds (int): Number of steps for intersection threshold.
        num_confidence_thresholds (int): Number of steps for confidence threshold.

    Returns:
        Dict[Tuple[float, float], Dict[str, float]]: A dictionary with threshold pairs
            as keys and corresponding global metrics as values.
    """
    global_threshold_metrics = {}
    for intersection_threshold in np.linspace(0.01, 1.0, num_intersection_thresholds):
        for confidence_threshold in np.linspace(0.24, 0.8, num_confidence_thresholds):
            all_actual_labels = []
            all_predicted_labels = []
            for _, one_video_data in filtered_detections_data.items():
                actual_labels = [
                    frame_data["label"] for frame_data in one_video_data.values()
                ]
                vehicle_presence_predictions = predict_vehicle_in_video(
                    one_video_data, intersection_threshold, confidence_threshold
                )
                predicted_labels = list(vehicle_presence_predictions.values())

                # Aggregate labels and predictions
                all_actual_labels.extend(actual_labels)
                all_predicted_labels.extend(predicted_labels)

            # Calculate global metrics
            calculated_metrics = calculate_metrics(
                all_actual_labels, all_predicted_labels  # type: ignore
            )
            global_threshold_metrics[
                (intersection_threshold, confidence_threshold)
            ] = calculated_metrics

    return global_threshold_metrics


def find_threshold(
    detections_file_path: str,
    vehicle_class_ids: List[int] = [0, 1, 2, 3, 4, 5, 7, 28],
) -> Tuple[Tuple[float, float], float]:
    """Finds the best intersection and confidence thresholds for vehicle detection based
        on the maximum F1-score.

    Args:
        detections_file_path (str): Path to JSON with raw detections for all videos.
        vehicle_class_ids (Set[int]): A set of class IDs to be considered as vehicles.

    Returns:
        Tuple[Optional[Tuple[float, float]], float]: A tuple containing the best
            threshold pair and the maximum F1-score.
    """
    # Reading the detection data
    with open(detections_file_path, "r") as file:
        detections_data = json.load(file)

    # Filter the detections data to include only vehicle objects
    filtered_detections_data = filter_vehicles(detections_data, vehicle_class_ids)
    global_threshold_metrics = calculate_global_metrics(filtered_detections_data)

    max_f1_score = 0.0
    best_threshold_key = (-1.0, -1.0)

    # Iterate through global metrics to find the best threshold based on max F1-score
    for threshold_key, video_metrics in tqdm(global_threshold_metrics.items()):
        f1 = video_metrics["f1_score"]
        if f1 > max_f1_score:  # Update if a higher F1-score is found
            max_f1_score = f1
            best_threshold_key = threshold_key
    print(
        f"Best thresholds, intersection {best_threshold_key[0]}, "
        f"confidence {best_threshold_key[1]}, "
        f"best f1 {max_f1_score}"
    )
    return best_threshold_key, max_f1_score


def validate_videos(
    detections_val_path: str,
    vehicle_class_ids: List[int] = [0, 1, 2, 3, 4, 5, 7, 28],
    intersection_threshold: float = 0.25,
    confidence_threshold: float = 0.3,
) -> Dict[str, float]:
    """Validates videos and calculates metrics

    Args:
        detections_val_path (str): Path to detections data for validation video.
        vehicle_class_ids (List[int], optional): A list of class IDs that are considered
            as vehicles. Defaults to [0, 1, 2, 3, 4, 5, 7, 28].
        intersection_threshold (float, optional): Minimum proportion of intersection to
            consider a detection significant. Defaults to 0.25
        confidence_threshold (float, optional): Minimum confidence level to consider a
            detection reliable. Defaults to 0.3
    Returns:
        Dict[str, float]: Return dictionary with metrics.
    """
    # read validation data
    # Reading the detection data
    with open(detections_val_path, "r") as file:
        detections_data = json.load(file)
    # filter vehicles
    filtered_detections_data = filter_vehicles(detections_data, vehicle_class_ids)
    video_data = list(filtered_detections_data.values())[0]
    # predict vehicles in video
    actual_labels = []
    predicted_labels = []
    for video_data in filtered_detections_data.values():
        predictions = predict_vehicle_in_video(
            video_data,
            intersection_threshold,
            confidence_threshold,
        )
        # prepare true and predict labels
        for frame, predict_label in predictions.items():
            actual_labels.append(video_data[frame]["label"])
            predicted_labels.append(predict_label)

    metrics = calculate_metrics(
        actual_labels,  # type: ignore
        predicted_labels,
    )

    return metrics


if __name__ == "__main__":
    # Create parser and initialize arguments
    parser = argparse.ArgumentParser(description="Find best thresolds.")
    parser.add_argument(
        "-v",
        "--video_to_val",
        nargs="+",
        default=None,
        help="Video name or list of video names, which was used for validation",
    )
    parser.add_argument(
        "-r",
        "--path_to_resources",
        default="resources",
        help="Path to the resources directory with intervals and polygons JSON files.",
    )

    # Collect arguments
    args = parser.parse_args()

    # Use collected arguments
    video_to_val = args.video_to_val
    path_to_resources = args.path_to_resources

    detection_dict_path = os.path.join(path_to_resources, "detections_dict.json")

    threshold_json_path = os.path.join(path_to_resources, "thresholds.json")
    thresholds_result = find_threshold(detection_dict_path)
    threshold_dict_to_save = {
        "intersection_threshold": thresholds_result[0][0],
        "confidence_threshold": thresholds_result[0][1],
    }
    # Save detection results to JSON
    with open(threshold_json_path, "w") as file:
        json.dump(threshold_dict_to_save, file)

    if video_to_val is not None:
        detection_dict_val_path = os.path.join(
            path_to_resources, "detections_dict_val.json"
        )
        validation_metrics = validate_videos(
            detections_val_path=detection_dict_val_path,
            intersection_threshold=thresholds_result[0][0],
            confidence_threshold=thresholds_result[0][1],
        )
        val_metrics_path = os.path.join(path_to_resources, "metrics_val.json")
        with open(val_metrics_path, "w") as file:
            json.dump(validation_metrics, file)
