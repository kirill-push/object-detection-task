# Data problems

## Video_0.mp4

In this video there are 39 frames and on all of them detector found 161 men, but all labels = 0 (no vehicles).
    (I understand that man it's not vehicle, but on some next videos man in frame will change label from 0 to 1)
Algo has errors on all frames.

## Video_1.mp4
In this video there are 18 frames, 9 with vehicle, 9 without. But detector find that only on 8 frames there are vehicles (all of them are trucks) and on frame 12 there are no any vehicles (see example).
Only on frame 12 algo has error.


## Video_2.mp4
In this video there are 8 frames with label 1 and 56 frames with label 0. Wrong prediciton on frames 45, 57, 60. Also there is small trolley in upper right corner on all frames, but it doesn't count for labeling. Also detector sometimes see trolley (but detect it like truck), but sometimes doesn't see. On frame 45 there is car on the upper rights edge of polygon (see example). On frame 57 detector is wrong (see bus, but intersection probably too small). On frame 60 detector doesn't see bus because it's behind the ladder


## Video_3.mp4
55 frames and 47 with false positive prediction.
On frame 0 - 2 there is a man. On frame 3 there are 2 men near truck on edge of polygon and their legs are hidden.
On frame 20 (and there are other frames) for example there is a man in polygon, but label 0.
On frame 43 with label 1 there is car on the upper edge of the polygon (on 2-45 it didn't count)


# Video_4.mp4
Similar video to video_3 - on the edge there are trucks.
On frame 20 - 52 on upper edge there is car, it seems like it is outside the polygon, but label 1. From frame 47 to frame 64 there are people


## Video_5.mp4
On last frames (96, 97) there people in polygon, but label is 0.


## Video_6.mp4
Frame 2 - label 0, but on edge inside polygon there is truck (see example).
Frame 109 - 119 - same situation, but on 109 label 1, but 110-119 label 0.
Frame 122 - truck on right edge - label 0, frame 123 - label 1.
Also there are people on last frames (with trucks)


## Video_7.mp4
On the majority of frames there people and it seems that there are no vehicles (there is one stationary object and may be it's truck, but on other video this kind of stationary object didn't count). On most of frames there are only people and stationary object.


## Video_8.mp4
This video is simmilar to video_7 (same place and time). But here on all frames label 0, but there is same stationary truck. For example frame 2.

