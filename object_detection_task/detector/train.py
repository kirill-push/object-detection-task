import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
        intersection_threshold (float): Minimum proportion of intersection to consider
            a detection significant.
        confidence_threshold (float): Minimum confidence level to consider a detection
            reliable.

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
