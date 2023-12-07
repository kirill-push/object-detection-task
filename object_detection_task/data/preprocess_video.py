import json
from typing import Dict, List

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
