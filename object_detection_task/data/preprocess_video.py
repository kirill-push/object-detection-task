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


def crop_polygon_from_frame(
    frame: np.ndarray,
    polygon: List[List[int]],
    same_size: bool = False,
    bg_color_id: Tuple[int, int, int] = (0, 0, 0),
    min_square: bool = False,
) -> np.ndarray:
    """Crops a polygon region from a given frame and optionally
        returns it within the bounds of the original frame size.

    Args:
        frame (np.ndarray): Frame from which polygon will be cropped.
        polygon (List[List[int]]): A list of [x, y] coordinates that define polygon.
        same_size (bool, optional): Whether to return the cropped region with
            the same dimensions as the original frame. Defaults to False.
        bg_color_id (Tuple[int, int, int], optional): Which color will be on background.
            Defaults to (0, 0, 0).
        min_square (bool, optional): If True - return croped frame in minimal square.
            Defaults to False.

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

    # Change background color
    if bg_color_id != (0, 0, 0):
        img2 = np.full(
            (result.shape[0], result.shape[1], 3), bg_color_id, dtype=np.uint8
        )
        mask_inv = cv2.bitwise_not(mask)
        color_crop = cv2.bitwise_or(img2, img2, mask=mask_inv)
        result = result + color_crop  # type: ignore

    if same_size:
        return result

    # Find the bounding rectangle of the polygon
    bound = cv2.boundingRect(points)  # returns (x, y, w, h) of the rect
    if min_square:
        return frame[bound[1] : bound[1] + bound[3], bound[0] : bound[0] + bound[2]]
    # Crop the frame to the bounding rectangle
    cropped = result[bound[1] : bound[1] + bound[3], bound[0] : bound[0] + bound[2]]
    return cropped
