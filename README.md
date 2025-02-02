Openpose light weight: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch?tab=readme-ov-file

Use openpose light weight because open pose doesn't seem to want to compile on mac... also we will need a lightweight application in any case. Still fairly slow on macbook though.

Vitpose: https://huggingface.co/docs/transformers/main/en/model_doc/vitpose

Use the huggingface version as it is easier to use. 

Mediapipe: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
Very optimized on CPU, not as accurate