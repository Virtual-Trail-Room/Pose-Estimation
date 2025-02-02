import torch
import requests
import numpy as np

from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

import math
import cv2

from vitpose_helper import *

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "https://media.istockphoto.com/id/1132930261/photo/full-length-body-size-side-profile-photo-jumping-high-beautiful-she-her-lady-hands-arms-up.jpg?s=612x612&w=0&k=20&c=n37aQSwx8IoEN7GX1iaf2y0-MskBRHtz2VxceScpBjE="
image = Image.open(requests.get(url, stream=True).raw)

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

keypoint_edges = model.config.edges


numpy_image = np.array(image)
person_boxes = detect_human(image, person_image_processor, person_model, device)
image_pose_result = detect_keypoints(image, person_boxes, image_processor, model, device)

for pose_result in image_pose_result:
    scores = np.array(pose_result["scores"])
    keypoints = np.array(pose_result["keypoints"])

    # draw each point on image
    draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)

    # draw links
    draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

opencv_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

cv2.imshow("pose estimation", opencv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()