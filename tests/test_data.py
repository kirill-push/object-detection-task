import json
from typing import List
from unittest.mock import MagicMock, mock_open, patch

import cv2
import numpy as np
import pytest

from object_detection_task.data import preprocess_video


def mock_video_capture(frames: List[np.ndarray]) -> MagicMock:
    """Creates a mock object for cv2.VideoCapture.

    Args:
        frames (List[np.ndarray]): A list of numpy arrays representing video frames.

    Returns:
        MagicMock: A mock video capture object with predefined read behavior.
    """
    video_capture = MagicMock()
    video_capture.isOpened.side_effect = [True] * (len(frames) + 1)
    video_capture.read.side_effect = [(True, frame) for frame in frames] + [
        (False, None)
    ]
    return video_capture


def test_extract_frames() -> None:
    # Create fake frames for testing
    fake_frames = [np.random.rand(100, 100, 3) for _ in range(5)]

    # Patch the cv2.VideoCapture class to return a mock video capture object
    with patch.object(
        cv2, "VideoCapture", return_value=mock_video_capture(fake_frames)
    ) as patched_capture:
        # Call the extract_frames function with a fake path
        extracted_frames = preprocess_video.extract_frames("fake_path")

    # Assert that the number of extracted frames is 5
    assert len(extracted_frames) == 5

    # Assert that cv2.VideoCapture was called with the fake path
    patched_capture.assert_called_with("fake_path")


@pytest.mark.parametrize("annotation", ["polygons", "intervals"])
def test_read_annotations(annotation: str) -> None:
    if annotation == "polygons":
        mock_data = {
            "video_0.mp4": [[536, 573], [873, 562], [824, 422]],
            "video_1.mp4": [[1, 2], [3, 4], [5, 6], [7, 100]],
        }
    else:
        mock_data = {
            "video_0.mp4": [],
            "video_1.mp4": [[1, 2], [3, 4]],
        }
    mock_json = json.dumps(mock_data)
    with patch("builtins.open", mock_open(read_data=mock_json)):
        annotations = preprocess_video.read_annotations("dummy_path.json")
    assert annotations == mock_data


def generate_test_frame(width: int, height: int) -> np.ndarray:
    """Test frame generation.

    Args:
        width (int): Width of generated frame.
        height (int): Height of generated frame.

    Returns:
        np.ndarray: Generated frame
    """
    return np.full((height, width, 3), 255, dtype=np.uint8)


@pytest.fixture
def mock_frame() -> np.ndarray:
    # Create a mock image (a white image with a black polygon)
    image_height = 100
    image_width = 100
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # White image
    polygon = np.array([[10, 10], [10, 50], [50, 50], [50, 10]])
    cv2.fillPoly(image, [polygon], (0, 0, 0))  # Fill the polygon with black
    return image


@pytest.mark.parametrize("same_size", [True, False])
def test_crop_rectangle_polygon(mock_frame: np.ndarray, same_size: bool) -> None:
    # mock_frame is not black
    assert not np.array_equal(mock_frame, np.zeros_like(mock_frame))
    polygon = [[10, 10], [10, 50], [50, 50], [50, 10]]
    cropped = preprocess_video.crop_polygon_from_frame(
        mock_frame, polygon, same_size=same_size
    )

    # Checking the size of the cropped image
    if same_size:
        assert cropped.shape == mock_frame.shape
        assert np.array_equal(cropped, np.zeros_like(mock_frame))
    else:
        assert cropped.shape == (41, 41, 3)


@pytest.mark.parametrize("same_size", [True, False])
def test_crop_irregular_polygon(mock_frame: np.ndarray, same_size: bool) -> None:
    polygon = [[10, 10], [20, 80], [30, 50], [80, 20]]
    cropped = preprocess_video.crop_polygon_from_frame(
        mock_frame, polygon, same_size=same_size
    )

    # Checking the size of the cropped image
    if same_size:
        assert cropped.shape == mock_frame.shape
    else:
        expected_height = 80 - 10 + 1
        expected_width = 80 - 10 + 1
        assert cropped.shape == (expected_height, expected_width, 3)
