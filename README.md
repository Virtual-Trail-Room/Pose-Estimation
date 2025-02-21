# Set up:

Set up environment (I use venv but anaconda should work as well)
```
$ python -m venv .venv
$ source .venv/bin/activate
```
Make sure to also change your python interpreter

Then download all necessary libraries:
```
$ pip install -r requirements .txt
```

# Model Comparisons
All the models are great at rendering video frame by frame when provided the prerecorded video as it can go frame by frame with no lag in final output. However, calculating landmarks and outputing a smooth image for real time use is much harder as the models need computing time. Table is made to help to compare model performances and other stats in real time application:

Note: all observations were done on MacBook Pro 2021 (no gpu support).

| Model       | FPS         | Accuracy      | Comments |
| :---        |    :----:   | :---:         | ---: | 
| Mediapipie  | 15          | low - medium  |  |
| Yolo        | 2-15        | medium - high | Versitile with multiple options |
| OpenPose    | 7           | high          |  |
| VitPose     | 2           | high          |  |


# [Openpose Light weight](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch?tab=readme-ov-file )

Use openpose light weight because open pose base code is poorly maintained; as result most of the repo is very unusuable. Results are very slow and frames appear to be chopped on Mac.

The pretrained model has been included in the repo already, but you can download it here as well: [pretrained model](https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth)

To run: 
```
$ cd openpose
$ python demo.py --checkpoint-path checkpoint_iter_370000.pth --video 0
```


# [Vitpose](https://huggingface.co/docs/transformers/main/en/model_doc/vitpose)

Hugging face provides a well maintained version model that is accessible through their `transformers` api, which is used in this code.

Provides very accurate pose estimation, however it doesn't handle cpu only support too well. Results are very slow and frames appear to be chopped on Mac. 

Run this `vitpose/vitpose_estimation.py` the way you would run a standard python.

# [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

Downloaded directly from MediaPipe API. Provides very fast cpu support, but a little less accurate.

Run this `Mediapipe/mp_pose_estimation.py` the way you would run a standard python.
 
# [Yolo](https://docs.ultralytics.com/tasks/pose/)

Using ultralytics api to download and use YOLO models locally. 

Run the file `Yolo/main.py` the way you would run a standard python file.

