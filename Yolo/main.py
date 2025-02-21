import cv2
import torch
import time
from ultralytics import YOLO

# Load YOLO Pose Model
model = YOLO('yolo11x-pose.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Define COCO keypoint connections for skeleton drawing
SKELETON_CONNECTIONS = [
    (1, 2), (1, 3), (2, 4), (3, 5),   # Face connections
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11),  # Arms
    (12, 13), (12, 14), (13, 15), (14, 16), (15, 17),
    (6, 12), (7, 13) # Legs
]

# Define colors for keypoints and skeleton
KEYPOINT_COLOR = (0, 255, 0)  # Green
SKELETON_COLOR = (255, 0, 0)  # Blue

# FPS Calculation
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO Pose Estimation
    results = model.track(frame, verbose=False)
    
    if results:
        for result in results:
            keypoints = result.keypoints.xy.cpu().numpy()

            for person in keypoints:
                valid_keypoints = [(int(x), int(y)) if x > 0 and y > 0 else None for x, y in person]

                for kp in valid_keypoints:
                    if kp:  # Only draw valid keypoints
                        cv2.circle(frame, kp, 5, KEYPOINT_COLOR, -1)

                for pt1, pt2 in SKELETON_CONNECTIONS:
                    if pt1 - 1 < len(valid_keypoints) and pt2 - 1 < len(valid_keypoints):
                        kp1, kp2 = valid_keypoints[pt1 - 1], valid_keypoints[pt2 - 1]
                        if kp1 and kp2:  # Only draw if both keypoints are valid
                            cv2.line(frame, kp1, kp2, SKELETON_COLOR, 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show result
    cv2.imshow('YOLO Pose Estimation', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
