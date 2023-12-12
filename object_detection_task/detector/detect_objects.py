import argparse
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from shapely.geometry import Polygon, box
from torch import nn
from tqdm import tqdm

from object_detection_task.data.preprocess_video import (
    AnnotationManager,
    VideoDataManager,
)


class ObjectDetector:
    def __init__(self, model_name: str = "yolov5x6"):
        self.model_name = model_name
        self.model = self.load_pretrained_yolov5(self.model_name)

    def load_pretrained_yolov5(self, model_name: str = "yolov5x6") -> nn.Module:
        """Load a pretrained YOLOv5 model.

        Args:
        model_name (str): Name of the YOLOv5 model to be loaded. Default is 'yolov5x6'.
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
        self,
        frame: np.ndarray,
        label: Optional[int] = None,
    ) -> List[Dict[str, Union[int, float]]]:
        """Detects objects in a given frame using a preloaded YOLOv5 model.

        Args:
            frame (np.ndarray): The prepared frame for object detection.
            label (int | None): If label is not None, than write label to detection dict.
                Defaults to None.

        Returns:
            List[Dict[str, Union[int, float]]]: A list of detected objects, each object is
                a dictionary containing coordinates ('x_min', 'y_min', 'x_max', 'y_max'),
                confidence score, and detected object class_id. Also dictionary contains
                label if label is not None.
        """

        # Perform detection
        results = self.model(frame)
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
        self,
        detections: List[Dict[str, Union[int, float]]],
        polygon: List[List[int]],
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

    def process_one_video(
        self,
        video_manager: VideoDataManager,
        target_size: Tuple[int, int] = (1280, 1280),
        up: int = 50,
        down: int = 50,
        left: int = 50,
        right: int = 50,
    ) -> Dict[str, Dict[str, List[Dict[str, Union[int, float]]]]]:
        """Processes all frames of one video, detecting objects and calculating
            intersections with a polygon.

        Args:
            model (torch.nn.Module): Loaded YOLOv5 model for object detection.
            video_manager (VideoDataManager): Video data manager for video
            target_size (Tuple[int, int]): Target frame size for the model.
                Defaults to (1280, 1280).
            up (int, optional): Padding on top. Defaults to 50.
            down (int, optional): Padding at bottom. Defaults to 50.
            right (int, optional): Padding on right. Defaults to 50.
            left (int, optional): Padding on left. Defaults to 50.

        Returns:
            Dict[str, Dict[str, List[Dict[str, Union[int, float]]]]]: Dictionary with
                n_frame as key and list of dictionaries as value. Each value list containe
                dictionaries with information about detections and intersections metrics.
        """
        # Label each frame
        frames_with_labels = video_manager.label_frames()

        all_frame_detections = {}
        for n_frame, (_, label) in tqdm(
            enumerate(frames_with_labels),
            total=len(frames_with_labels),
            desc=f"Processing frames from {video_manager.video_name}",
        ):
            prepared_frame = video_manager.prepare_frame_for_detector(
                n_frame=n_frame,
                target_size=target_size,
                up=up,
                down=down,
                left=left,
                right=right,
            )
            prepared_polygon = video_manager.recalculate_polygon_coordinates(
                target_size=target_size,
                up=up,
                down=down,
                left=left,
                right=right,
            )
            # Detect objects in the frame
            detections = self.detect_objects(prepared_frame, label)
            if detections:
                # Check intersections and calculate metrics for each detection
                frame_detections_with_metrics = self.check_intersection(
                    detections, prepared_polygon
                )
            else:
                frame_detections_with_metrics = []
            all_frame_detections[n_frame] = {
                "label": label,
                "detector_result": frame_detections_with_metrics,
            }

        return all_frame_detections

    def process_all_video(
        self,
        video_list: Optional[List[str]],
        intervals_data_path: str,
        polygons_data_path: str,
        video_dir_path: str = "resources/videos",
    ) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
        """Processes all frames of all videos, detecting objects and calculating
            intersections with a polygon.

        Args:
            video_list (List[str] | None): List of videos to process. If None, then use 
            all videos.
            intervals_data_path (str): Path to intevals annotation.
            polygons_data_path (str): Path to polygons annotation.
            video_dir_path (str): Path to directory with videos.

        Returns:
            Dict[str, Dict[str, Dict[str, List[Dict]]]]: _description_
        """
        all_detections = {}
        if video_list is None:
            video_list = AnnotationManager(polygons_data_path, "polygons").video_list
        for video in video_list:
            video_path = os.path.join(video_dir_path, video)
            video_manager = VideoDataManager(
                video_path, intervals_data_path, polygons_data_path
            )
            processed_frames = self.process_one_video(
                video_manager=video_manager,
            )
            all_detections[video] = processed_frames
        return all_detections


if __name__ == "__main__":
    # Create parser and initialize arguments
    parser = argparse.ArgumentParser(description="Detect objects on videos")
    parser.add_argument("--video_to_val", default=None, help="Name of video to validate")
    parser.add_argument(
        "--path_to_resources", default="resources", help="Path to the resources directory"
    )

    # Collect arguments
    args = parser.parse_args()

    # Use collected arguments
    video_to_val = args.video_to_val
    path_to_resources = args.path_to_resources

    intervals_data_path = os.path.join(path_to_resources, "time_intervals.json")
    polygons_data_path = os.path.join(path_to_resources, "polygons.json")
    video_dir_path = os.path.join(path_to_resources, "videos")
    video_list = [  # TODO use train_test_split
        video
        for video in AnnotationManager(polygons_data_path, "polygons").video_list
        if video != video_to_val
    ]
    detector = ObjectDetector()
    # Process videos and detect objects
    detections = detector.process_all_video(
        video_list=video_list,
        intervals_data_path=intervals_data_path,
        polygons_data_path=polygons_data_path,
        video_dir_path=video_dir_path,
    )

    # Save detection results to JSON
    output = os.path.join(path_to_resources, f'detections_dict.json')
    with open(output, "w") as file:
        json.dump(detections, file)

    if video_to_val is not None:
        detections_val = detector.process_all_video(
            video_list=[video_to_val],
            intervals_data_path=intervals_data_path,
            polygons_data_path=polygons_data_path,
            video_dir_path=video_dir_path,
        )
        # Save detection results to JSON
        output_val = os.path.join(path_to_resources, f"detections_dict_{video_to_val.split('.')[0]}.json")
        with open(output_val, "w") as file:
            json.dump(detections_val, file)