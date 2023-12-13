[![Linting](https://github.com/kirill-push/object-detection-task/actions/workflows/lint.yml/badge.svg)](https://github.com/kirill-push/object-detection-task/actions/workflows/lint.yml)
---
# Important information
## Script run_detector.sh
You can execute [run_detector.sh](run_detector.sh) with custom paths to video, polygon and output.
Polygon file should contain dictionary the same as original `polygons.json` file.
```
   ./run_detector.sh video_path polygon_path output_path
```

## Work Analysis Report
You can find Work Analysis Report in [English](REPORT_eng.md) and [Russian](REPORT_ru.md).


# Table of Contents
0. [Important information](#Important-information)
1. [Brief task description](#Brief-task-description)
2. [Activate Poetry Environment](#activate-poetry-environment)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Install Poetry](#step-2-install-poetry)
   - [Step 3: Install Dependencies Using Poetry](#step-3-install-dependencies-using-poetry)
   - [Step 4: Verify Your Setup](#step-4-verify-your-setup)
3. [Baseline](#baseline)
   - [Running the evaluate_baseline.py Script](#running-the-evaluate_baselinepy-script)
4. [Detector](#detector)
   - [Running the run_detector.py Script](#running-the-run_detectorpy-script)
   - [Running the calculate_metrics.py Script](#running-the-calculate_metricspy-script)
   - [Running the detect_objects.py Script](#running-the-detect_objectspy-script)
   - [Running the train.py Script](#running-the-trainpy-script)
5. [Fixing Problems with Project Initialization](#fixing-problems-with-project-initialization)
   - [Fixing OpenCV cv2 ImportError on Linux](#fixing-opencv-cv2-importerror-on-linux)


# Brief task description
Fixed cameras have been installed to monitor the area where the aircraft is preparing for departure. The task is to make sure that there are no vehicles on the stand before the aircraft arrives. If any vehicles are found inside the booth boundary, the ground operator should be notified.

This repository offers a baseline solution with [Pixel Brightness variance Analysis](#baseline), as well as a solution with [Object Detection and the selection of treshold](#detector).

# Activate Poetry Environment

## Step 1: Clone the Repository

1. First, clone the project repository from its source. This can usually be done using a command like:
   ```
   git clone https://github.com/kirill-push/object-detection-task.git
   ```

2. After cloning, navigate into the project directory:
   ```
   cd object-detection-task
   ```

## Step 2: Install Poetry

If you don't have Poetry installed, you'll need to install it. I recommend using Poetry version 1.6 or higher, but any version above 1.2 should suffice.

1. To install Poetry, run:
   ```
   pipx install poetry
   ```
   or
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Alternatively, you can visit [Poetry's installation guide](https://python-poetry.org/docs/#installation) for other methods.

2. Verify the installation with:
   ```
   poetry --version
   ```

## Step 3: Install Dependencies Using Poetry

With Poetry installed, you can now install the project's dependencies.

1. To install dependencies, run from project directory:
   ```
   poetry install
   ```

   This command reads the `pyproject.toml` file and installs all the dependencies listed there.

## Step 4: Verify Your Setup

1. Check that everything is set up correctly by running a simple command, like:
   ```
   poetry run python --version
   ```

   This should display your Python version, which should be 3.8 or higher.

# Baseline
## Running the evaluate_baseline.py

1. **Activate Poetry Environment**: Ensure you are in the Poetry-managed virtual environment by running:
   ```
   poetry shell
   ```

2. **Running the Script**: The `evaluate_baseline.py` script in the `baseline` directory accepts the following arguments:

   - `-v` or `--video_to_val`: One video name or list of video names to validate.
   - `-d` or `--path_to_video_dir`: Path to the directory with all videos. Default is `"resources/videos"`.
   - `-i` or `--file_path_intervals`: Path to intervals annotation. Default is `"resources/time_intervals.json"`.
   - `-p` or `--file_path_polygons`: Path to polygons annotation. Default is `"resources/polygons.json"`.
   - `-r` or `--path_to_resources`: Path to the resources directory.

   To run the script, use a command in the following format:
   ```
   python baseline/evaluate_baseline.py [-v video_1.mp4 video_2.mp4 ...] [-d path/to/video/dir] [-i path/to/intervals.json] [-p path/to/polygons.json] [-r path/to/resources]
   ```

   Example:
   ```
   python baseline/evaluate_baseline.py -v video_1.mp4 video_2.mp4 -d resources/videos -i resources/time_intervals.json -p resources/polygons.json -r resources
   ```

   This command will run the baseline evaluation on the specified videos using the resources from the given path.

3. **Output**: The script will evaluate the baseline on the videos and output the results in JSON format in the `baseline_metrics_test.json` file within the specified resources directory.

### Notes:

- The script will automatically exclude `video_16.mp4` and `video_17.mp4` as they are duplicates of `video_4.mp4` and `video_3.mp4`, respectively.
- If no videos are specified for validation (`--video_to_val`), the script will process all videos except for the excluded and specified validation videos.
- The resources directory should contain `time_intervals.json`, `polygons.json`, and a `videos` subdirectory with the video files.


# Detector
## Running the run_detector.py Script

1. **Activate Poetry Environment**: Activate the Poetry virtual environment by running:
   ```
   poetry shell
   ```

2. **Run the Script**: The `run_detector.py` script accepts the following arguments:

   - `-v` or `--video_path`: Path(s) to one or more videos to process. This argument is required.
   - `-p` or `--polygon_path`: Path to the JSON file containing boundaries for the videos. This argument is required.
   - `-o` or `--output_path`: Path to save the results in a JSON file. This argument is required.
   - `-t` or `--thresholds_path`: Path to the JSON file with thresholds. Default is `"resources/thresholds.json"`.

   To run the script, use a command in the following format:
   ```
   python object_detection_task/detector/run_detector.py -v path/to/video.mp4 -p path/to/polygon.json -o path/to/save/results.json [-t path/to/thresholds.json]
   ```

   Example:
   ```
   python object_detection_task/detector/run_detector.py -v resources/videos/video_1.mp4 resources/videos/video_2.mp4 -p resources/polygons.json -o resources/val_intervals.json
   ```

   This command will process the specified videos, using the given polygon boundaries and thresholds, and save the predicted intervals in the specified output JSON file.

3. **Output**: The script will predict the labels for each video and save the intervals in the specified output file.

### Notes:

- Ensure that the paths to the videos, polygon JSON file, and output file are correctly specified.
- The script allows processing multiple videos in one run. You can list several video paths separated by spaces.
- The default thresholds path assumes that the `thresholds.json` file is in the `resources` directory. You can specify a different path if necessary.


## Running the calculate_metrics.py Script

1. **Activate Poetry Environment**: Start the Poetry virtual environment:
   ```
   poetry shell
   ```

2. **Run the Script**: The `calculate_metrics.py` script requires several arguments:

   - `-v` or `--video_path`: Path to the video file for which you want to calculate metrics.
   - `-i` or `--intervals_path`: Path to the JSON file containing intervals for the video.
   - `-p` or `--polygons_path`: Path to the JSON file containing boundaries for the video.
   - `-o` or `--output_path`: Path where the results will be saved in a JSON file.
   - `-t` or `--thresholds_path`: Path to the JSON file containing thresholds (optional, defaults to `"resources/thresholds.json"`).

   To run the script, use a command in the following format:
   ```
   python object_detection_task/metrics/calculate_metrics.py -v path/to/video.mp4 -i path/to/intervals.json -p path/to/polygons.json -o path/to/output.json [-t path/to/thresholds.json]
   ```

   Example:
   ```
   python object_detection_task/metrics/calculate_metrics.py -v resources/videos/video_1.mp4 -i resources/time_intervals.json -p resources/polygons.json -o resources/metrics_video_1.json
   ```

   This command will calculate the metrics for the specified video using the provided intervals and polygons, and then save the results in the specified output file.

3. **Output**: The script will generate metrics for the video and save them in the specified output file.

### Notes:

- Ensure that the paths to the video, intervals JSON file, polygons JSON file, output file, and thresholds (if not using default) are correctly specified.
- The script is designed to work with one video at a time.


## Running the detect_objects.py Script

1. **Activate Poetry Environment**: Ensure you are in the Poetry-managed virtual environment by running:
   ```
   poetry shell
   ```

2. **Running the Script**: The `detect_objects.py` script has two primary arguments:

   - `-v` or `--video_to_val`: One video name or list of video names to validate.
   - `-r` or `--path_to_resources`: Path to the resources directory.

   To run the script, use a command in the following format:
   ```
   python object_detection_task/detector/detect_objects.py [-v video_1.mp4 video_2.mp4 ...] [-r path/to/resources]
   ```

   Example:
   ```
   python object_detection_task/detector/detect_objects.py -v video_1.mp4 video_2.mp4 -r resources
   ```

   This command will run the object detection on the specified videos using the resources from the given path.

3. **Output**: The script will process the videos and output detection results in JSON format in the specified resources directory and save it with name detections_dict.json.

### Notes:

- The script will ignore `video_16.mp4` and `video_17.mp4` as they are duplicates of `video_4.mp4` and `video_3.mp4`, respectively.
- If no videos are specified for validation (`--video_to_val`), the script will process all videos except the ignored and specified validation videos.
- The resources directory should contain `time_intervals.json`, `polygons.json`, and a `videos` subdirectory with the video files.


## Running the train.py Script

1. **Activate Poetry Environment**: Activate the Poetry virtual environment by running:
   ```
   poetry shell
   ```

2. **Running the Script**: The `train.py` script accepts the following arguments:

   - `-v` or `--video_to_val`: Specifies one or more video names used for validation. This is optional.
   - `-r` or `--path_to_resources`: The path to the resources directory containing intervals and polygons JSON files. The default is `"resources"`.

   To run the script, use a command in the following format:
   ```
   python object_detection_task/detector/train.py [-v video_1.mp4 video_2.mp4 ...] [-r path/to/resources]
   ```

   Example:
   ```
   python object_detection_task/detector/train.py -v video_1.mp4 video_2.mp4 -r resources
   ```

   This will start the training process using the specified videos for validation and resources from the given path.

3. **Output**: The script performs the following operations:
   - Finds the best thresholds based on the detections in `detections_dict.json`.
   - Saves the calculated thresholds in `thresholds.json` within the resources directory.
   - If validation videos are provided, it validates these videos using the calculated thresholds and saves the metrics in `detector_metrics_test.json`.

### Notes:

- Make sure the `resources` directory contains `detections_dict.json` and `detections_dict_val.json` (if validation videos are specified).
- The script saves the threshold values and validation metrics in JSON format in the specified resources directory.


---

# Fixing problems with project initialization
## Fixing OpenCV `cv2` ImportError on Linux

If you encounter the following error while using the `cv2` module in the OpenCV library:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

This indicates that the necessary OpenGL libraries are missing from your environment. Follow these steps to resolve the issue:

### For Ubuntu/Debian-based Systems:

1. **Update Package List**: Update your system's package list to ensure you can access the latest versions of packages.

   ```bash
   sudo apt-get update
   ```

2. **Install OpenGL Dependency**: Install the `libgl1` package which provides the `libGL.so.1` library.

   ```bash
   sudo apt-get install libgl1
   ```


## Resolving `ModuleNotFoundError: No module named 'pkg_resources'` in Python

If you encounter the following error in Python:

```
ModuleNotFoundError: No module named 'pkg_resources'
```

This error typically occurs due to the absence of `setuptools` or an outdated version in your environment. Follow these steps to resolve the issue:

### Installing or Updating `setuptools`:

1. **Install `setuptools`**: If you don't have `setuptools` installed, you can install it using `pip`. It's also a good idea to upgrade it to the latest version.

   ```
   pip install --upgrade setuptools
   ```

---
