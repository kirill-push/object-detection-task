#!/bin/bash

DEFAULT_VIDEO_PATH="resources/videos/video_5.mp4"
DEFAULT_POLYGON_PATH="resources/polygons.json"
DEFAULT_OUTPUT_PATH="resources/test_intervals.json"

VIDEO_PATH=${1:-$DEFAULT_VIDEO_PATH}
POLYGON_PATH=${2:-$DEFAULT_POLYGON_PATH}
OUTPUT_PATH=${3:-$DEFAULT_OUTPUT_PATH}

python object_detection_task/detector/run_detector.py -v $VIDEO_PATH -p $POLYGON_PATH -o $OUTPUT_PATH
