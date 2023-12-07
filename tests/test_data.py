from typing import List
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

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
