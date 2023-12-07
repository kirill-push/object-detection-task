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


def crop_polygon_from_frame(
    frame: np.ndarray, polygon: List[List[int]], same_size: bool = False
) -> np.ndarray:
    """Crops a polygon region from a given frame and optionally
        returns it within the bounds of the original frame size.

    Args:
        frame (np.ndarray): Frame from which polygon will be cropped.
        polygon (List[List[int]]): A list of [x, y] coordinates that define polygon.
        same_size (bool, optional): Whether to return the cropped region with
            the same dimensions as the original frame. Defaults to False.

    Returns:
        np.ndarray: The cropped region of the frame, either as a masked image with
            the same size as the original frame or as the smallest bounding rectangle
            around the polygon.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array(polygon)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, pts=[points], color=(255, 255, 255))

    # Apply the mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=mask)
    if same_size:
        return result

    # Find the bounding rectangle of the polygon
    bound = cv2.boundingRect(points)  # returns (x, y, w, h) of the rect

    # Crop the frame to the bounding rectangle
    cropped = result[bound[1] : bound[1] + bound[3], bound[0] : bound[0] + bound[2]]
    return cropped
