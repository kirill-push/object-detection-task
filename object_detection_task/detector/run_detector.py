import argparse
import json
from typing import Dict, List

from object_detection_task.data.preprocess_video import VideoDataManager
from object_detection_task.detector.detect_objects import ObjectDetector
from object_detection_task.detector.train import predict_vehicle_in_video


def predict(
    video_manager: VideoDataManager,
    thresholds_path: str = "resources/thresholds.json",
) -> Dict[str, int]:
    """Makes oredictions for one video.

    Args:
        video_manager (VideoDataManager): Video data manager for predict video.
        thresholds_path (str): Path to JSON file with thresholds.

    Returns:
        Dict[str, int]: A dict containing frames as keys and prediction as values.
    """
    # Reading thresholds dict
    with open(thresholds_path, "r") as file:
        thresholds_dict = json.load(file)
    intersection_threshold = thresholds_dict["intersection_threshold"]
    confidence_threshold = thresholds_dict["confidence_threshold"]

    detector = ObjectDetector()

    # Detect objects on video
    frame_detection = detector.process_one_video(
        video_manager=video_manager,
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
        if predictions[frame] == 1 and start_frame is None:
            # Start of a new interval
            start_frame = int(frame)
        elif predictions[frame] == 0 and start_frame is not None:
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
    parser.add_argument(
        "-v",
        "--video_path",
        nargs="+",
        help="Path to one or few videos which we want to process",
    )
    parser.add_argument(
        "-p", "--polygon_path", help="Path to JSON with boundaries for this video"
    )
    parser.add_argument("-o", "--output_path", help="Path to JSON file to save results")
    parser.add_argument(
        "-t,",
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

    if isinstance(video_path, str):
        video_path = [video_path]
    intervals = {}
    for one_video_path in video_path:
        video_manager = VideoDataManager(one_video_path, None, polygon_path)
        video_name = video_manager.video_name
        one_video_intervals = make_intervals(
            predict(
                video_manager=video_manager,
                thresholds_path=thresholds_path,
            )
        )
        intervals[video_name] = one_video_intervals

    # Save intervals to JSON
    with open(output_path, "w") as file:
        json.dump(intervals, file)
