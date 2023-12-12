import json
from unittest.mock import mock_open, patch

import pytest

from object_detection_task.data.preprocess_video import (
    AnnotationManager,
)


@pytest.mark.parametrize("mode", ["polygons", "intervals"])
def test_annotations_manager(mode: str) -> None:
    if mode == "polygons":
        mock_data = {
            "video_0.mp4": [[536, 573], [873, 562], [824, 422]],
            "video_1.mp4": [[1, 2], [3, 4], [5, 6], [7, 100]],
        }
    else:
        mock_data = {
            "video_0.mp4": [],
            "video_1.mp4": [[1, 4], [6, 6], [9, 10]],
        }
    mock_json = json.dumps(mock_data)
    # test read_annotations()
    with patch("builtins.open", mock_open(read_data=mock_json)):
        annotations = AnnotationManager("dummy_path.json", mode)
    assert annotations.data() == mock_data
    # test get_labels
    test_length = 12
    if mode == "intervals":
        labels = annotations.get_labels("video_1.mp4", test_length)
        assert labels == [0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
