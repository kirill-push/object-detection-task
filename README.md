Sure, here's a sample instruction guide for starting a project using Poetry:

---

# Getting Started with Project

## Step 1: Clone the Repository

1. First, clone the project repository from its source. This can usually be done using a command like:
   ```
   git clone https://github.com/kirill-push/object-detection-task.git
   ```

2. After cloning, navigate into the project directory:
   ```
   cd object_detection_task
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


## Running the detect_objects.py Script

1. **Activate Poetry Environment**: Ensure you are in the Poetry-managed virtual environment by running:
   ```
   poetry shell
   ```

2. **Navigate to the Detector Directory**: Change to the `detector` directory where the `detect_objects.py` script is located:
   ```
   cd detector
   ```

3. **Running the Script**: The `detect_objects.py` script has two primary arguments:

   - `-v` or `--video_to_val`: One video name or list of video names to validate.
   - `-p` or `--path_to_resources`: Path to the resources directory.

   To run the script, use a command in the following format:
   ```
   python detect_objects.py [-v video_1.mp4 video_2.mp4 ...] [-p path/to/resources]
   ```

   Example:
   ```
   python detect_objects.py -v video_1.mp4 video_2.mp4 -p resources
   ```

   This command will run the object detection on the specified videos using the resources from the given path.

4. **Output**: The script will process the videos and output detection results in JSON format in the specified resources directory and save it with name detections_dict.json.

### Notes:

- The script will ignore `video_16.mp4` and `video_17.mp4` as they are duplicates of `video_4.mp4` and `video_3.mp4`, respectively.
- If no videos are specified for validation (`--video_to_val`), the script will process all videos except the ignored and specified validation videos.
- The resources directory should contain `time_intervals.json`, `polygons.json`, and a `videos` subdirectory with the video files.


## Running the train.py Script

1. **Activate Poetry Environment**: Activate the Poetry virtual environment by running:
   ```
   poetry shell
   ```

2. **Navigate to the Detector Directory**: Change to the `detector` directory where the `train.py` script is located:
   ```
   cd detector
   ```

3. **Running the Script**: The `train.py` script accepts the following arguments:

   - `-v` or `--video_to_val`: Specifies one or more video names used for validation. This is optional.
   - `-r` or `--path_to_resources`: The path to the resources directory containing intervals and polygons JSON files. The default is `"resources"`.

   To run the script, use a command in the following format:
   ```
   poetry run python train.py [-v video_1.mp4 video_2.mp4 ...] [-p path/to/resources]
   ```

   Example:
   ```
   poetry run python train.py -v video_1.mp4 video_2.mp4 -p resources
   ```

   This will start the training process using the specified videos for validation and resources from the given path.

4. **Output**: The script performs the following operations:
   - Finds the best thresholds based on the detections in `detections_dict.json`.
   - Saves the calculated thresholds in `thresholds.json` within the resources directory.
   - If validation videos are provided, it validates these videos using the calculated thresholds and saves the metrics in `metrics_val.json`.

### Notes:

- Make sure the `resources` directory contains `detections_dict.json` and `detections_dict_val.json` (if validation videos are specified).
- The script saves the threshold values and validation metrics in JSON format in the specified resources directory.


## Running the run_detector.py Script

1. **Activate Poetry Environment**: Activate the Poetry virtual environment by running:
   ```
   poetry shell
   ```

2. **Navigate to the Detector Directory**: Change to the `detector` directory where the `run_detector.py` script is located:
   ```
   cd detector
   ```

3. **Run the Script**: The `run_detector.py` script accepts the following arguments:

   - `-v` or `--video_path`: Path(s) to one or more videos to process. This argument is required.
   - `-p` or `--polygon_path`: Path to the JSON file containing boundaries for the videos. This argument is required.
   - `-o` or `--output_path`: Path to save the results in a JSON file. This argument is required.
   - `-t` or `--thresholds_path`: Path to the JSON file with thresholds. Default is `"resources/thresholds.json"`.

   To run the script, use a command in the following format:
   ```
   poetry run python run_detector.py --video_path path/to/video.mp4 --polygon_path path/to/polygon.json --output_path path/to/save/results.json [--thresholds_path path/to/thresholds.json]
   ```

   Example:
   ```
   poetry run python run_detector.py -v resources/videos/video_1.mp4 resources/videos/video_2.mp4 -p resources/polygon.json -o resources/val_intervals.json
   ```

   This command will process the specified videos, using the given polygon boundaries and thresholds, and save the predicted intervals in the specified output JSON file.

4. **Output**: The script will predict the labels for each video and save the intervals in the specified output file.

### Notes:

- Ensure that the paths to the videos, polygon JSON file, and output file are correctly specified.
- The script allows processing multiple videos in one run. You can list several video paths separated by spaces.
- The default thresholds path assumes that the `thresholds.json` file is in the `resources` directory. You can specify a different path if necessary.


## Running the calculate_metrics.py Script

1. **Activate Poetry Environment**: Start the Poetry virtual environment:
   ```
   poetry shell
   ```

2. **Navigate to the Metrics Directory**: Change to the `metrics` directory where the `calculate_metrics.py` script is located:
   ```
   cd metrics
   ```

3. **Run the Script**: The `calculate_metrics.py` script requires several arguments:

   - `-v` or `--video_path`: Path to the video file for which you want to calculate metrics.
   - `-i` or `--intervals_path`: Path to the JSON file containing intervals for the video.
   - `-p` or `--polygons_path`: Path to the JSON file containing boundaries for the video.
   - `-o` or `--output_path`: Path where the results will be saved in a JSON file.
   - `-t` or `--thresholds_path`: Path to the JSON file containing thresholds (optional, defaults to `"resources/thresholds.json"`).

   To run the script, use a command in the following format:
   ```
   poetry run python calculate_metrics.py -v path/to/video.mp4 -i path/to/intervals.json -p path/to/polygons.json -o path/to/output.json [-t path/to/thresholds.json]
   ```

   Example:
   ```
   poetry run python calculate_metrics.py -v resources/videos/video.mp4 -i resources/time_intervals.json -p resources/polygons.json -o resources/metrics_val.json
   ```

   This command will calculate the metrics for the specified video using the provided intervals and polygons, and then save the results in the specified output file.

4. **Output**: The script will generate metrics for the video and save them in the specified output file.

### Notes:

- Ensure that the paths to the video, intervals JSON file, polygons JSON file, output file, and thresholds (if not using default) are correctly specified.
- The script is designed to work with one video at a time.
