import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
import cv2
from vitpose_helper import *

device = "cuda" if torch.cuda.is_available() else "cpu"

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

keypoint_edges = model.config.edges

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    cv_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(cv_image_rgb)

    try: 
        person_boxes = detect_human(image, person_image_processor, person_model, device)
        image_pose_result = detect_keypoints(image, person_boxes, image_processor, model, device)

        for pose_result in image_pose_result:
            scores = np.array(pose_result["scores"])
            keypoints = np.array(pose_result["keypoints"])
            draw_points(frame, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)
            draw_links(frame, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

        cv2.imshow("pose estimation", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    except:
        print("no person found")

cv2.destroyAllWindows()