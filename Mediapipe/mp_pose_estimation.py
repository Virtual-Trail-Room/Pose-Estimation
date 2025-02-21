import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
prev_time = time.time()
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
      ret, frame = cap.read()

      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False
      
      results = pose.process(image)

      image.flags.writeable = True

      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      curr_time = time.time()
      fps = 1 / (curr_time - prev_time)
      prev_time = curr_time
      cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


      cv2.imshow("mediapipe feed", image)
      if cv2.waitKey(10) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()