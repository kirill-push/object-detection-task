import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def extract_frames(video_path: str) -> List[np.ndarray]:
    """Extracts all frames from the given video file.

    Args:
        video_path (str): Path to the video file from which frames will be extracted.

    Returns:
        List[np.ndarray]: List of numpy arrays representing frames from the video.
    """
    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)

    video.release()
    return frames


def read_annotations(annotation_path: str) -> Dict[str, List[List[int]]]:
    """Reads and parses annotations from a JSON file.

    Args:
        annotation_path (str): The path to the JSON file containing annotations.

    Returns:
        Dict[str, List[List[int]]]: A dictionary where keys are video names and values
        are lists of lists, each list representing:
        - for time_interval.json: video frame intervals when vehicle was in polygon;
        - for polygons.json: — coordinates of pixels of polygon: [x, y].
    """
    with open(annotation_path, "r") as file:
        annotations = json.load(file)
    return annotations


def label_frames(
    frames: List[np.ndarray], annotations: Dict[str, List[List[int]]], video_name: str
) -> List[Tuple[np.ndarray, int]]:
    """Labels each frame based on the presence of a car within the annotated intervals.

    Args:
        frames (List[np.ndarray]): A list of frames from a video.
        annotations (Dict[str, List[List[int]]]): A dictionary containing intervals
            for each video where a car is present.
        video_name (str): The name of the video for which frames are being labeled.

    Returns:
        List[Tuple[np.ndarray, int]]: A list of tuples where each tuple contains a
            frame and its corresponding label (1 for car present, 0 for no car).
    """
    if video_name not in annotations:
        raise ValueError("Wrong video name, check annotations")
    labeled_frames = []
    for i, frame in enumerate(frames):
        label = 0  # Default label (no car)
        for interval in annotations[video_name]:
            if interval[0] <= i <= interval[1]:
                label = 1  # Car is present
                break
        labeled_frames.append((frame, label))
    return labeled_frames
