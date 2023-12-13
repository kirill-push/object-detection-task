# Work Analysis Report

## Results and Analysis

### a. Final Results on Test and Training Sets

- **Overall Results:** 

  [Object Detector Model](#b.-Technology-Choices-for-Solution)
  
  - **F1 Score:** 0.759
  - **Recall:** 0.961
  - **Precision:** 0.628

  [Detector Metrics file](resources/detector_metrics_test.json)

- **Individual Video Results:**
  Files [video_3.mp4](resources/video_3_metrics.json), [video_5.mp4](resources/video_5_metrics.json), [video_7.mp4](resources/video_7_metrics.json)

- **Test File Selection:** Simple video with constant objects on the site (video 7), simple video with changing objects on the site (video 5), video with changing objects in the area of the polygon, but also with constant objects at the edges of the polygon (video 3). Videos 16 and 17 were also removed from training because they duplicate videos 3 and 4.

- **Baseline Method Results:**

  [Baseline model](#b.-Technology-Choices-for-Solution)

  - **F1 Score:** 0.616
  - **Recall:** 0.690
  - **Precision:** 0.557

  [Baseline Metrics file](resources/baseline_metrics_test.json)

### Results Interpretation

1. **Overall Results:** 
   - A high **Recall** (0.961) indicates that the algorithm finds objects of the first class well. However, moderate **Precision** (0.628) indicates a significant number of false positives. **F1 Score** (0.759), balancing Recall and Precision, is also good, but not perfect due to these trade-offs.

2. **Comparison with [Baseline Method](#b.-Technology-Choices-for-Solution):** 
   - The detector algorithm outperforms the baseline method in all metrics, especially in higher **Recall** and **F1 Score**.

3. **Need to know:**
   - Despite acceptable test results, it should be noted that the algorithm performs poorly under some conditions described below. If you specifically choose test videos, you can get poor results.

**i. Metrics Used and Reasons for Choice**

- **Recall:** This metric was chosen because it shows how well the algorithm detects Class 1 objects. The lower the Recall, the greater the risk of not detecting a vehicle in the polygon area (higher False Positive), which can lead to an accident.
- **Precision:** Needed to reduce False Positive errors. Low precision leads to false alarms, causing, for example, dispatcher dissatisfaction or unnecessary checks.
- **F1 Score:** Used to balance between Recall and Precision, especially important in data imbalance conditions.
- **Accuracy:** Not used due to data imbalance.

**ii. Solution Limitations**

- **Polygon Edge Object Detecting:** The model struggles with objects at the edges of the polygon.
- **Airport Specifics:** The detector wasn't further trained on data, sometimes missing, for example, towing machines uncommon in everyday life.
- **Online Processing:** The algorithm needs adaptation for frame-by-frame real-time online processing.

**iii. Script for Running Metric Calculation**

```
python object_detection_task/metrics/calculate_metrics.py -v resources/videos/video_test.mp4 -i resources/time_intervals.json -p resources/polygons.json -o resources/metrics_video_test.json
```

### b. Technology Choices for Solution

#### **Baseline (Brightness Variance):**
Explored as a baseline approach. Frames cropped closer to the size of the polygon are converted into brightness variances. Then, for each video, results are normalized to the standard average and a threshold is chosen that maximizes the F1 metric. This approach was chosen for its simplicity and quick implementation, but it was less effective than the detector algorithm.
#### **Object Detector and Threshold bruteforce for Intersection with Polygon:**
Using a pre-trained detector, objects are found in each frame. Then, the relative areas of intersection of each object's bounding box with the polygon are calculated. Thresholds for the relative area and the detector's confidence in the found object are then selected. Thresholds are chosen based on the maximum F1 metric. YOLOv5 was chosen as the detector for its simplicity and high efficiency.

### c. Github repository with python code which process video

This repository: [https://github.com/kirill-push/object-detection-task](https://github.com/kirill-push/object-detection-task)

### d. The script on python3 which is processing video file and save results into JSON

#### Script run_detector.py

For full instructions how to run run_detector.py file click [here](README.md#Running-the-run_detector.py-Script).

Brief:
```
   python object_detection_task/detector/run_detector.py -v path/to/video.mp4 -p path/to/polygon.json -o path/to/save/results.json [-t path/to/thresholds.json]
```

#### Script run_detector.sh
You can execute [run_detector.sh](run_detector.sh) with custom paths to video, polygon and output.
Polygon file should contain dictionary the same as original `polygons.json` file.
```
   ./run_detector.sh video_path polygon_path output_path
```

## Recommendations for Improvement

1. **Improving Labeling:** Conduct a detailed analysis of the algorithm's errors, perhaps some data should be relabeled to eliminate errors (errors definitely exist).
2. **Increasing Test and Training Sample Sizes:** To be more confident in the results (and to use more advanced solutions), the sample size should be increased.
3. **Integrating Frame Sequences:** Use sequence information to increase accuracy.
4. **Model Retraining:** Adapt the detector to detect objects typical of airports, retrain the detector. Possibly use training of the pre-trained detector backbone with a decision-making head, which will take a frame as an input, and return the probability of object in the polygon at the output.
5. **Combining Detector with Tracking:** Using tracking can improve detection results.
6. **Applying Segmentation:** For more accurate determination of object intersection with the polygon.
7. **Code testing:** Write tests that cover the entire code and consider all possible ways to use the model.

## Conclusions

This solution shows good results on the test sample, but has a number of limitations related to the quality of data labeling, processing of objects at the boundaries of polygons, features of vehicles in the airport, and not using information about the sequence of video frames. Further work should focus on improving the model and experimenting with more complex solutions.
