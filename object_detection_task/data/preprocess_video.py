import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class AnnotationManager:
    def __init__(self, annotation_path: str, mode: str):
        self.annotation_path = annotation_path
        self.annotations = self.read_annotations()
        if mode not in ["polygons", "intervals"]:
            raise ValueError("Wrong mode, should be polygons or intervals")
        self.mode = mode
        self.video_list = list(self.annotations.keys())

    def read_annotations(self) -> Dict[str, List[List[int]]]:
        """Reads and parses annotations from a JSON file.

        Returns:
            Dict[str, List[List[int]]]: Dictionary where keys are video names and values
            are lists of lists, each list representing:
            - for time_interval.json: video frame intervals when vehicle was in polygon;
            - for polygons.json: — coordinates of pixels of polygon: [x, y].
        """
        with open(self.annotation_path, "r") as file:
            annotations = json.load(file)
        if not isinstance(annotations, Dict):
            raise ValueError("Annotations should store dict with key video_name")
        return annotations

    def get_labels(self, video_name: str, video_length: int) -> List[int]:
        """Get labels list from intervals

        Args:
            video_name (str): Video name.
            video_length (int): Video length.

        Returns:
            List[int]: List of labels with 0 or 1 according to intervals.
        """
        if self.mode != "intervals":
            raise ValueError(
                "Wrong annotation manager, initialize it with mode 'intervals"
            )
        labels = [0] * video_length
        intervals = self(video_name)
        for interval in intervals:
            start, end = interval
            for i in range(start, end + 1):
                labels[i] = 1
        return labels

    def data(self) -> Dict[str, List[List[int]]]:
        """Returns annotations dictionary.

        Returns:
            Dict[str, List[List[int]]]: Annotations dictionary.
        """
        return self.annotations

    def __call__(self, video_name: str) -> List[List[int]]:
        """Return annotation for video_name.

        Args:
            video_name (str): Name of video.

        Returns:
            List[List[int]]: Annotation for video_name.
        """
        return self.annotations[video_name]


class VideoDataManager:
    def __init__(
        self,
        video_path: str,
        intervals_path: Optional[str]='../resources/time_intervals.json',
        polygons_path: str = '../resources/polygons.json'
    ):
        self.video_path = video_path
        self.video_dir_path = os.path.dirname(video_path)
        self.video_name = os.path.basename(video_path)

        self.frames = self.extract_frames()
        self.length = len(self.frames)
        if intervals_path is not None:
            self.intervals_manager = AnnotationManager(intervals_path, "intervals")
            self.intervals = self.intervals_manager(self.video_name)
        else:
            self.intervals = None  # type: ignore

        self.polygons_manager = AnnotationManager(polygons_path, "polygons")
        self.polygon = self.polygons_manager(self.video_name)
        if intervals_path is not None:
            self.labels = self.intervals_manager.get_labels(
                self.video_name, self.length
            )
        else:
            self.labels = [-1] * self.length

    def extract_frames(self) -> List[np.ndarray]:
        """Extracts all frames from the given video file.

        Returns:
            List[np.ndarray]: List of numpy arrays representing frames from the video.
        """
        video = cv2.VideoCapture(self.video_path)
        frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frames.append(frame)

        video.release()
        return frames

    def prepare_frame_for_detector(
        self,
        n_frame: int,
        target_size: Tuple[int, int] = (1280, 1280),
        keep_aspect_ratio: bool = True,
        up: int = 50,
        down: int = 50,
        right: int = 50,
        left: int = 50,
    ) -> np.ndarray:
        """Prepares video frame for detector processing.

        Args:
            n_frame (int): Number of frame.
            target_size (Tuple[int, int]): Target frame size for the model.
                Defaults to (1280, 1280).
            keep_aspect_ratio (bool): If True, keeps the aspect ratio while scaling.
                Defaults to True.
            up (int, optional): Padding on top. Defaults to 0.
            down (int, optional): Padding at bottom. Defaults to 0.
            right (int, optional): Padding on right. Defaults to 0.
            left (int, optional): Padding on left. Defaults to 0.

        Returns:
            np.ndarray: The prepared frame as a numpy array.
        """
        cropped_frame = self.crop_polygon_from_frame(
            n_frame=n_frame,
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

    def crop_polygon_from_frame(
        self,
        n_frame: int,
        same_size: bool = False,
        bg_color_id: Tuple[int, int, int] = (0, 0, 0),
        min_square: bool = False,
        up: int = 50,
        down: int = 50,
        right: int = 50,
        left: int = 50,
    ) -> np.ndarray:
        """Crops a polygon region from a given frame with additional padding.

        Args:
            n_frame (int): Number of frame from which polygon will be cropped.
            same_size (bool, optional): Whether to return the cropped region with
                the same dimensions as the original frame. Defaults to False.
            bg_color_id (Tuple[int, int, int], optional): Which color will be on
                background. Defaults to (0, 0, 0).
            min_square (bool, optional): If True - return cropped frame in minimal
                square. Defaults to False.
            up (int, optional): Padding on top. Defaults to 50.
            down (int, optional): Padding at bottom. Defaults to 50.
            right (int, optional): Padding on right. Defaults to 50.
            left (int, optional): Padding on left. Defaults to 50.

        Returns:
            np.ndarray: The cropped region of the frame with specified paddings.
        """
        frame = self.frames[n_frame]
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = np.array(self.polygon)

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
        w, h = (
            min(frame.shape[1] - x, w + right + left),
            min(frame.shape[0] - y, h + up + down),
        )

        if min_square:
            return frame[y : y + h, x : x + w]

        # Crop the frame to the adjusted bounding rectangle
        cropped = result[y : y + h, x : x + w]
        return cropped

    def label_frames(self) -> List[Tuple[np.ndarray, int]]:
        """Labels each frame based on the presence of a car within the annotated
            intervals.

        Returns:
            List[Tuple[np.ndarray, int]]: A list of tuples where each tuple contains a
                frame and its corresponding label (1 for car present, 0 for no car).
                If interval is None, then return List[Tuple[np.ndarray, -1]]
        """
        self.labeled_frames = []
        for i, frame in enumerate(self.frames):
            label = 0  # Default label (no car)
            if self.intervals is not None:
                for interval in self.intervals:
                    if interval[0] <= i <= interval[1]:
                        label = 1  # Car is present
                        break
            else:
                label = -1  # type: ignore
            self.labeled_frames.append((frame, label))
        return self.labeled_frames

    def recalculate_polygon_coordinates(
        self,
        target_size: Tuple[int, int] = (1280, 1280),
        keep_aspect_ratio: bool = True,
        up: int = 50,
        down: int = 50,
        right: int = 50,
        left: int = 50,
    ) -> List[List[int]]:
        """Recalculates coordinates of polygon after frame has been cropped and resized.

        Args:
            target_size (Tuple[int, int]): Target size of frame after resizing.
            keep_aspect_ratio (bool): If True, keeps the aspect ratio while resizing.
            up (int, optional): Padding on top. Defaults to 50.
            down (int, optional): Padding at bottom. Defaults to 50.
            right (int, optional): Padding on right. Defaults to 50.
            left (int, optional): Padding on left. Defaults to 50.

        Returns:
            List[List[int]]: The recalculated polygon coordinates.
        """
        original_polygon = self.polygon
        original_frame_size = self.frames[0].shape[:2]
        # Find the bounding rectangle of the polygon
        points = np.array(original_polygon)
        x0, y0, w0, h0 = cv2.boundingRect(points)
        x0, y0 = max(0, x0 - left), max(0, y0 - up)
        w, h = (
            min(original_frame_size[1] - x0, w0 + right + left),
            min(original_frame_size[0] - y0, h0 + up + down),
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
        self.recalculated_polygon = []
        for point in new_polygon:
            new_x = int(point[0] * scale[0])
            new_y = int(point[1] * scale[1])
            self.recalculated_polygon.append([new_x, new_y])

        return self.recalculated_polygon

    def draw_polygon_on_frame(
        self,
        n_frame: int,
        crop: bool = True,
        inplace: bool = False,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        bbox: Optional[List[int]] = None,
        bbox_color: Tuple[int, int, int] = (0, 0, 255),
        bbox_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws a polygon on the given frame.

        Args:
            nframe (int): Number of frame on which to draw the polygon.
            crop (bool, optional): If crop is True, then returns cropped frame.
                Defaults to True.
            inplace (bool, optional): If inplace is True, then save result to
                self.frames. Defaults to False.
            color (Tuple[int, int, int]): The color of the polygon in BGR format.
                Defaults to (0, 255, 0) (green).
            thickness (int): The thickness of the polygon lines.
                Defaults to 2.
            bbox (List[int] | None): If bbox is not None, then draw it.
                Coordinates of bbox = [x_min, y_min, x_max, y_max].
                Defaults to None.
            bbox_color (Tuple[int, int, int]): Color of the bounding box in BGR format.
                Defaults to (0, 0, 255) (red).
            bbox_thickness (int): Thickness of the bounding box lines.
                Defaults to 1.
        Returns:
            np.ndarray: The frame with the polygon and optionally a bounding box drawn
                on it.
        """
        if crop:
            polygon = self.recalculate_polygon_coordinates()
            frame = self.crop_polygon_from_frame(n_frame)
        else:
            polygon = self.polygon
            frame = self.frames[n_frame]
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        new_frame = frame.copy()

        # Draw the polygon on the frame
        cv2.polylines(new_frame, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw the bbox if provided
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(
                new_frame, (x_min, y_min), (x_max, y_max), bbox_color, bbox_thickness
            )
        if inplace:
            self.frames[n_frame] = new_frame
        return new_frame


def safe_dict_to_json(any_dict: Dict, path_to_file: str) -> None:
    """Save dictionary to JSON

    Args:
        any_dict (Dict): Any dictionary.
        path_to_file (str): Saving path.
    """
    with open(path_to_file, "w") as file:
        json.dump(any_dict, file)
