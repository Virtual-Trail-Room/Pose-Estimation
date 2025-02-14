import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model trained for pose estimation
model = YOLO('yolo11s-pose.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(frame)
    # results = model.track(frame, show=True)
    
    # manually drawing them is much faster than using built in show=True
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()
        for person in keypoints:
            for x, y in person:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                
    cv2.imshow('Pose Estimation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
