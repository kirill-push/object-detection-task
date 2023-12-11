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


