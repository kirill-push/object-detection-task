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


def label_frames(  # TODO: input only interval for current video, without video_name
    frames: List[np.ndarray],
    intervals_data: Dict[str, List[List[int]]],
    video_name: str,
) -> List[Tuple[np.ndarray, int]]:
    """Labels each frame based on the presence of a car within the annotated intervals.

    Args:
        frames (List[np.ndarray]): A list of frames from a video.
        intervals_data (Dict[str, List[List[int]]]): A dictionary containing intervals
            for each video where a car is present.
        video_name (str): The name of the video for which frames are being labeled.

    Returns:
        List[Tuple[np.ndarray, int]]: A list of tuples where each tuple contains a
            frame and its corresponding label (1 for car present, 0 for no car).
    """
    if video_name not in intervals_data:
        raise ValueError("Wrong video name, check intervals_data")
    labeled_frames = []
    for i, frame in enumerate(frames):
        label = 0  # Default label (no car)
        for interval in intervals_data[video_name]:
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
    up: int = 0,
    down: int = 0,
    right: int = 0,
    left: int = 0,
) -> np.ndarray:
    """Crops a polygon region from a given frame with additional padding.

    Args:
        frame (np.ndarray): Frame from which polygon will be cropped.
        polygon (List[List[int]]): A list of [x, y] coordinates that define polygon.
        same_size (bool, optional): Whether to return the cropped region with
            the same dimensions as the original frame. Defaults to False.
        bg_color_id (Tuple[int, int, int], optional): Which color will be on background.
            Defaults to (0, 0, 0).
        min_square (bool, optional): If True - return cropped frame in minimal square.
            Defaults to False.
        up (int, optional): Padding on top. Defaults to 0.
        down (int, optional): Padding at bottom. Defaults to 0.
        right (int, optional): Padding on right. Defaults to 0.
        left (int, optional): Padding on left. Defaults to 0.

    Returns:
        np.ndarray: The cropped region of the frame with specified paddings.
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

    # Find the bounding rectangle of the polygon and add padding
    x, y, w, h = cv2.boundingRect(points)
    x, y = max(0, x - left), max(0, y - up)
    w, h = min(frame.shape[1] - x, w + right + left), min(
        frame.shape[0] - y, h + up + down
    )

    if min_square:
        return frame[y : y + h, x : x + w]

    # Crop the frame to the adjusted bounding rectangle
    cropped = result[y : y + h, x : x + w]
    return cropped


def prepare_frame_for_detector(
    frame: np.ndarray,
    polygon: List[List[int]],
    target_size: Tuple[int, int] = (640, 640),
    keep_aspect_ratio: bool = True,
    up: int = 0,
    down: int = 0,
    right: int = 0,
    left: int = 0,
) -> np.ndarray:
    """Prepares video frame for detector processing, including scaling and conversion.

    Args:
        frame (np.ndarray): The original video frame as a numpy array.
        polygons (List[List[int]]): A list of [x, y] coordinates that define polygon.
        target_size (Tuple[int, int]): Target frame size for the model.
            Defaults to (640, 640).
        keep_aspect_ratio (bool): If True, keeps the aspect ratio while scaling.
            Defaults to True.
        up (int, optional): Padding on top. Defaults to 0.
        down (int, optional): Padding at bottom. Defaults to 0.
        right (int, optional): Padding on right. Defaults to 0.
        left (int, optional): Padding on left. Defaults to 0.

    Returns:
        np.ndarray: The prepared frame as a numpy array.
    """

    # Crop the frame to the minimum rectangle enclosing the polygon
    cropped_frame = crop_polygon_from_frame(
        frame,
        polygon,
        min_square=True,
        up=up,
        down=down,
        right=right,
        left=left,
    )

    if keep_aspect_ratio:
        # Calculate new dimensions, keeping the aspect ratio
        h, w = cropped_frame.shape[:2]
        r = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * r)
        new_h = int(h * r)

        resized_frame = cv2.resize(
            cropped_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        # Create a new frame of the desired size and place the resized frame in it
        new_frame = np.full(
            (target_size[1], target_size[0], 3), (0, 0, 0), dtype=np.uint8
        )
        new_frame[:new_h, :new_w] = resized_frame
    else:
        # Simply resize the frame to the desired size
        new_frame = cv2.resize(
            cropped_frame, target_size, interpolation=cv2.INTER_LINEAR
        )

    return new_frame


def recalculate_polygon_coordinates(
    original_polygon: List[List[int]],
    original_frame_size: Tuple[int, int],
    target_size: Tuple[int, int] = (640, 640),
    keep_aspect_ratio: bool = True,
    up: int = 0,
    down: int = 0,
    right: int = 0,
    left: int = 0,
) -> List[List[int]]:
    """Recalculates coordinates of a polygon after frame has been cropped and resized.

    Args:
        original_polygon (List[List[int]]): Original polygon coordinates [[x, y], ...].
        original_frame_size (Tuple[int, int]): Original size of frame before cropping.
        target_size (Tuple[int, int]): Target size of frame after resizing.
        keep_aspect_ratio (bool): If True, keeps the aspect ratio while resizing.
        up (int, optional): Padding on top. Defaults to 0.
        down (int, optional): Padding at bottom. Defaults to 0.
        right (int, optional): Padding on right. Defaults to 0.
        left (int, optional): Padding on left. Defaults to 0.

    Returns:
        List[List[int]]: The recalculated polygon coordinates.
    """
    # Find the bounding rectangle of the polygon
    points = np.array(original_polygon)
    x0, y0, w0, h0 = cv2.boundingRect(points)
    x0, y0 = max(0, x0 - left), max(0, y0 - up)
    w, h = min(original_frame_size[1] - x0, w0 + right + left), min(
        original_frame_size[0] - y0, h0 + up + down
    )

    # Recalculate coordinates to be relative to the bounding rectangle
    new_polygon = [[x - x0, y - y0] for [x, y] in original_polygon]
    # Calculate the scale factors
    if keep_aspect_ratio:
        r = min(target_size[0] / w, target_size[1] / h)
        scale = (r, r)
    else:
        scale = (target_size[0] / w, target_size[1] / h)

    # Recalculate polygon coordinates
    recalculated_polygon = []
    for point in new_polygon:
        new_x = int(point[0] * scale[0])
        new_y = int(point[1] * scale[1])
        recalculated_polygon.append([new_x, new_y])

    return recalculated_polygon
