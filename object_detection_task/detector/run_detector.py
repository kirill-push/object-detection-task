import argparse
import json
import os
from typing import Dict, List

from object_detection_task.data.preprocess_video import read_annotations
from object_detection_task.detector.detect_objects import (
    load_pretrained_yolov5,
    process_one_video,
)
from object_detection_task.detector.train import predict_vehicle_in_video


def predict(
    video_path: str,
    polygons_path: str,
    thresholds_path: str = "resources/thresholds.json",
) -> Dict[str, int]:
    """Makes oredictions for one video.

    Args:
        video_path (str): Path to video which we want to process.
        polygons_path (str): Path to JSON with boundaries for this video.
        thresholds_path (str): Path to JSON file with thresholds.

    Returns:
        Dict[str, int]: A dict containing frames as keys and prediction as values.
    """
    # Reading thresholds dict
    with open(thresholds_path, "r") as file:
        thresholds_dict = json.load(file)
    intersection_threshold = thresholds_dict["intersection_threshold"]
    confidence_threshold = thresholds_dict["confidence_threshold"]

    # Init model
    model = load_pretrained_yolov5("yolov5x6")

    # Prepare args to process func
    video_dir_path = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    polygons_data = read_annotations(polygons_path)
    if isinstance(polygons_data, Dict):
        polygon = polygons_data[video_name]
    else:
        raise ValueError("polygons.json should store dict with key video_name")

    # Detect objects on video
    frame_detection = process_one_video(
        model=model,
        video_name=video_name,
        video_dir_path=video_dir_path,
        polygon=polygon,
        intervals=None,
    )

    # Process detector data to prediction
    predictions = predict_vehicle_in_video(
        video_data=frame_detection,
        intersection_threshold=intersection_threshold,
        confidence_threshold=confidence_threshold,
    )
    return predictions


def make_intervals(predictions: Dict[str, int]) -> List[List[int]]:
    """Makes intervals from prediction.

    Args:
        predictions (Dict[str, int]): Dictionary with frame numbers and predictions.

    Returns:
        List[List[int]]: List with intervals.
    """
    intervals = []
    start_frame = None

    # Convert keys to integers for correct sorting
    sorted_keys = sorted(predictions.keys(), key=lambda x: int(x))

    for frame in sorted_keys:
        if predictions[str(frame)] == 1 and start_frame is None:
            # Start of a new interval
            start_frame = int(frame)
        elif predictions[str(frame)] == 0 and start_frame is not None:
            # End of the current interval
            intervals.append([start_frame, int(frame) - 1])
            start_frame = None

    # Check if the video ends during an interval
    if start_frame is not None:
        intervals.append([start_frame, int(sorted_keys[-1])])
    return intervals


if __name__ == "__main__":
    # Create parser and initialize arguments
    parser = argparse.ArgumentParser(description="Predict intervals from video")
    parser.add_argument("--video_path", help="Path to video which we want to process")
    parser.add_argument(
        "--polygon_path", help="Path to JSON with boundaries for this video"
    )
    parser.add_argument("--output_path", help="Path to JSON file to save results")
    parser.add_argument(
        "--thresholds_path",
        default="resources/thresholds.json",
        help="Path to JSON file to save results",
    )

    # Collect arguments
    args = parser.parse_args()

    # Use collected arguments
    video_path = args.video_path
    polygon_path = args.polygon_path
    output_path = args.output_path
    thresholds_path = args.thresholds_path

    intervals = make_intervals(
        predict(
            video_path=video_path,
            polygons_path=polygon_path,
            thresholds_path=thresholds_path,
        )
    )

    # Save intervals to JSON
    with open(output_path, "w") as file:
        json.dump(intervals, file)
