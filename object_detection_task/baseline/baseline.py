import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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


def normalize_frame_variance(
    variance_dict: Dict[int, Tuple[float, int]]
) -> Dict[int, Tuple[float, int]]:
    """Normalize variance values of video frames using z-score normalization,
        keeping the labels unchanged.

    Args:
    variance_dict (dict): A dictionary where keys are frame numbers and values are
        tuples of the variance values and labels.

    Returns:
    Dict[int, Tuple[float, int]]: A dictionary with normalized variance values and
        unchanged labels, where keys are frame numbers.
    """

    # Extract variance values from the dictionary
    variance_values = np.array([value[0] for value in variance_dict.values()])

    # Calculate the mean (mu) and standard deviation (sigma) of the variance values
    mu = np.mean(variance_values)
    sigma = np.std(variance_values)

    # Normalize the variance values using z-score formula: (x - mu) / sigma
    normalized_variance = (variance_values - mu) / sigma

    # Reconstruct the dictionary with normalized values and unchanged labels
    normalized_dict = {
        key: (normalized_value, variance_dict[key][1])
        for key, normalized_value in zip(variance_dict.keys(), normalized_variance)
    }

    return normalized_dict


def visualize_variance_data(variance_dict: Dict[int, Tuple[float, int]]) -> None:
    """Visualize variances distribution for each label using histograms and box plots.

    Args:
    variance_dict (dict): A dictionary where keys are frame numbers and values are
        tuples of normalized variance values and labels.
    """

    # Separating the variance values based on labels
    variance_label_0 = [value[0] for value in variance_dict.values() if value[1] == 0]
    variance_label_1 = [value[0] for value in variance_dict.values() if value[1] == 1]

    # Creating histograms for each label
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(variance_label_0, bins=20, alpha=0.5, label="Label 0")
    plt.hist(variance_label_1, bins=20, alpha=0.5, label="Label 1")
    plt.title("Histogram of variance Values")
    plt.xlabel("variance Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Creating box plots for each label
    plt.subplot(1, 2, 2)
    plt.boxplot([variance_label_0, variance_label_1], labels=["Label 0", "Label 1"])
    plt.title("Box Plot of variance Values")
    plt.ylabel("variance Value")

    plt.tight_layout()
    plt.show()


def process_all_videos(
    path_to_video_dir: str,
    file_path_intervals: str,
    file_path_polygons: str,
    video_list: Optional[List[str]] = None,
    min_square: bool = True,
) -> Dict[Tuple[str, int], Tuple[float, int]]:
    """Process all videos in the list to calculate brightness variances,
        label them, and visualize the combined variance data.

    Args:
        path_to_video_dir (str): The video dir path.
        file_path_intervals (str): The file path for interval annotations.
        file_path_polygons (str): The file path for polygon annotations.
        video_list (List[str] | None): A list of video names.
            Defaults to None.
        min_square (bool, optional): Flag to determine cropping method.
            Defaults to True.

    Returns:
        Dict[Tuple[str, int], Tuple[float, int]]: Dictionary with combined data from all
            videos from video_list (if videos_list is None, than from all videos) with:
                key: (video name, frame number);
                value: (frame normalized variance, label).
    """

    combined_normalized_variance_dict = {}
    if video_list is None:
        video_list = list(read_annotations(file_path_polygons).keys())
    for video_name in video_list:
        video_path = os.path.join(path_to_video_dir, video_name)
        variance_dict = analyze_video_brightness_variance(
            video_path,
            file_path_intervals,
            file_path_polygons,
            min_square,
        )

        # Normalize the variance for the current video
        normalized_variance_dict = normalize_frame_variance(variance_dict)

        # Combine the normalized data from all videos
        for key, value in normalized_variance_dict.items():
            combined_normalized_variance_dict[(video_name, key)] = value

    return combined_normalized_variance_dict
