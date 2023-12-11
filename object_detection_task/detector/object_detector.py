import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from shapely.geometry import Polygon, box
from torch import nn
from tqdm import tqdm

from object_detection_task.data.preprocess_video import (
    extract_frames,
    label_frames,
    prepare_frame_for_detector,
    recalculate_polygon_coordinates,
)


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

    # Load the pretrained model from the official YOLOv5 repository
    model = torch.hub.load(
        "ultralytics/yolov5",
        model_name,
        force_reload=True,
        _verbose=False,
    )  # type: ignore

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


def process_one_video_frames(
    model: torch.nn.Module,
    video_name: str,
    video_dir_path: str,
    polygon: List[List[int]],
    intervals: List[List[int]],
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Processes all frames of one video, detecting objects and calculating
        intersections with a polygon.

    Args:
        model (torch.nn.Module): Loaded YOLOv5 model for object detection.
        video_name (str): Name of video.
        video_dir_path (str): Path to directory with videos.
        polygon (List[List[int]]): Polygon for video_name.
        intervals (List[List[int]]): Intervals for video_name.

    Returns:
        Dict[int, List[Dict[str, Union[int, float]]]]: A dictionary with n_frame as key
            and list of dictionaries as value. Each value list containe dictionaries
            with information about detections and intersections metrics.
    """
    up = 50
    down = 50
    left = 50
    right = 50

    # Define full path to the video.
    video_path = os.path.join(video_dir_path, video_name)

    # Extract frames from the video
    frames = extract_frames(video_path)
    # Label each frame
    frames_with_labels = label_frames(frames=frames, intervals=intervals)

    all_frame_detections = {}
    for n_frame, (frame, label) in tqdm(
        enumerate(frames_with_labels),
        total=len(frames_with_labels),
        desc="Processing frames",
    ):
        prepared_frame = prepare_frame_for_detector(
            frame=frame,
            polygon=polygon,
            up=up,
            down=down,
            left=left,
            right=right,
        )
        prepared_polygon = recalculate_polygon_coordinates(
            polygon,
            frame.shape[:2],  # type: ignore
            up=up,
            down=down,
            left=left,
            right=right,
        )
        # Detect objects in the frame
        detections = detect_objects(prepared_frame, model, label)
        if detections:
            # Check intersections and calculate metrics for each detection
            frame_detections_with_metrics = check_intersection(
                detections, prepared_polygon
            )
        else:
            frame_detections_with_metrics = []
        all_frame_detections[n_frame] = {
            'label': label,
            'detector_result': frame_detections_with_metrics
        }

    return all_frame_detections
