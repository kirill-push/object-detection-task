from typing import Dict, Tuple

import numpy as np

from object_detection_task.data.preprocess_video import (
    crop_polygon_from_frame,
    extract_frames,
    label_frames,
    read_annotations,
)


def calculate_brightness(frame: np.ndarray) -> np.ndarray:
    """Calculate the brightness of each pixel in an image frame.

    Args:
        frame (np.ndarray): A numpy array representing an image frame.
            Array shape should be (height, width, 3).

    Returns:
        np.ndarray: A 2D numpy array of the same height and width as the input frame,
            containing the brightness values of each pixel.
    """
    # Normalizing the color values to be between 0 and 1
    normalized_frame = frame / 255.0

    # Calculating the brightness for each pixel
    brightness = np.sum(normalized_frame**2, axis=2) / 3
    return brightness


def calculate_brightness_variance(cropped_frame: np.ndarray) -> float:
    """Calculates the variance of brightness in a cropped frame.

    Args:
        cropped_frame (np.ndarray): The cropped region of the frame.

    Returns:
        float: The variance of the brightness in the cropped frame.
    """
    brightness = calculate_brightness(cropped_frame)
    variance = np.var(brightness)
    return variance  # type: ignore


def analyze_video_brightness_variance(
    video_path: str,
    file_path_intervals: str,
    file_path_polygons: str,
    min_square: bool = True,
) -> Dict[int, Tuple[float, int]]:
    """
    Analyze the brightness variance in a video.

    Args:
        video_path (str): The path to the video file.
        file_path_intervals (str): The path to the file containing interval annotations.
        file_path_polygons (str): The path to the file containing polygon annotations.
        min_square (bool, optional): Flag to determine cropping method.
            Defaults to True.

    Returns:
        Dict[int, Tuple[float, int]]: A dictionary mapping frame indices to a tuple of
            variance and label.
    """

    video_name = video_path.split("/")[-1]
    frames = extract_frames(video_path)
    intervals_annotations = read_annotations(file_path_intervals)
    polygon_annotations = read_annotations(file_path_polygons)
    polygon = polygon_annotations[video_name]

    labeled_frames = label_frames(frames, intervals_annotations, video_name)
    variance_dict = {}

    # Loop through each frame, crop it using the polygon,
    # and calculate its brightness variance
    for i, (frame, label) in enumerate(labeled_frames):
        cropped = crop_polygon_from_frame(frame, polygon, min_square=min_square)
        variance = calculate_brightness_variance(cropped)
        variance_dict[i] = (variance, label)

    return variance_dict


def normalize_frame_dispersion(
    dispersion_dict: Dict[int, Tuple[float, int]]
) -> Dict[int, Tuple[float, int]]:
    """Normalize dispersion values of video frames using z-score normalization,
        keeping the labels unchanged.

    Args:
    dispersion_dict (dict): A dictionary where keys are frame numbers and values are
        tuples of the dispersion values and labels.

    Returns:
    Dict[int, Tuple[float, int]]: A dictionary with normalized dispersion values and
        unchanged labels, where keys are frame numbers.
    """

    # Extract dispersion values from the dictionary
    dispersion_values = np.array([value[0] for value in dispersion_dict.values()])

    # Calculate the mean (mu) and standard deviation (sigma) of the dispersion values
    mu = np.mean(dispersion_values)
    sigma = np.std(dispersion_values)

    # Normalize the dispersion values using z-score formula: (x - mu) / sigma
    normalized_dispersion = (dispersion_values - mu) / sigma

    # Reconstruct the dictionary with normalized values and unchanged labels
    normalized_dict = {
        key: (normalized_value, dispersion_dict[key][1])
        for key, normalized_value in zip(dispersion_dict.keys(), normalized_dispersion)
    }

    return normalized_dict
