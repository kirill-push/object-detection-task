from typing import Dict, List, Optional, Union

import numpy as np
import torch
from shapely.geometry import Polygon, box
from torch import nn


def load_pretrained_yolov5(model_name: str = "yolov5s") -> nn.Module:
    """Load a pretrained YOLOv5 model.

    Args:
    model_name (str): Name of the YOLOv5 model to be loaded. Default is 'yolov5s'.
        Available models are 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'.
        The 's', 'm', 'l', and 'x' variants represent small, medium, large, and
        extra-large models respectively.

    Returns:
    nn.Module: A PyTorch YOLOv5 model loaded with pretrained weights.
    """
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # URL for the YOLOv5 GitHub repository
    github_repo_url = "ultralytics/yolov5"

    # Load the pretrained model from the official YOLOv5 repository
    model = torch.hub.load(github_repo_url, model_name, pretrained=True)  # type: ignore

    return model.eval().to(device)


def detect_objects(
    frame: np.ndarray,
    model: nn.Module,
    label: Optional[int] = None,
) -> List[Dict[str, Union[int, float]]]:
    """Detects objects in a given frame using a preloaded YOLOv5 model.

    Args:
        frame (np.ndarray): The prepared frame for object detection.
        model (torch.nn.Module): The preloaded YOLOv5 model.
        label (int | None): If label is not None, than write label to detection dict.
            Defaults to None.

    Returns:
        List[Dict[str, Union[int, float]]]: A list of detected objects, each object is
            a dictionary containing coordinates ('x_min', 'y_min', 'x_max', 'y_max'),
            confidence score, and detected object class_id. Also dictionary contains
            label if label is not None.
    """

    # Perform detection
    results = model(frame)
    # Process results
    detections = []
    for detection in results.xyxy[0].cpu().numpy():
        x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
        one_detection_dict = {
            "x_min": int(x_min),
            "y_min": int(y_min),
            "x_max": int(x_max),
            "y_max": int(y_max),
            "confidence": float(confidence),
            "class_id": int(class_id),
        }
        if label is not None:
            one_detection_dict["label"] = label
        detections.append(one_detection_dict)
    return detections


def check_intersection(
    detections: List[Dict[str, Union[int, float]]], polygon: List[List[int]]
) -> List[Dict[str, Union[int, float]]]:
    """Determines whether a detected objects intersect with a given polygon and
        calculates metrics.

    Args:
        detections (List[Dict[str, Union[int, float]]]): List with dictionaries
            containing detection information with keys 'x_min', 'y_min', 'x_max',
            'y_max', 'confidence' and 'class_id' for one frame.
        polygon (List[List[int]]): List of [x, y] coordinates representing the polygon.

    Returns:
        List[Dict[str, Union[int, float]]]: List with dictionaries containing
            information about detection and intersections metrics.
    """
    # Create a polygon object from the list of points
    poly = Polygon(polygon)
    result_detections = detections.copy()
    list_to_del = []
    for i, detection in enumerate(result_detections):
        # Create a rectangular polygon from the detection coordinates
        bbox = box(
            detection["x_min"],
            detection["y_min"],
            detection["x_max"],
            detection["y_max"],
        )
        detection["intersect"] = poly.intersects(bbox)

        # Calculate the intersection of the detection and the polygon
        intersection = poly.intersection(bbox)
        if detection["intersect"]:
            # Calculate metrics
            detection["intersection_area"] = intersection.area
            detection["proportion_of_intersection"] = (
                detection["intersection_area"] / bbox.area
            )
        else:
            # Add to list number of object without intersection
            list_to_del.append(i)
    # Delete object dict without intersection to Polygon
    for i in list_to_del[::-1]:
        result_detections.pop(i)

    return result_detections
