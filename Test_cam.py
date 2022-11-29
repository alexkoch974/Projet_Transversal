import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

FACE = [67, 103, 54, 297, 332, 284, 10, 152, 234, 454, 473, 474, 475, 476, 477, 468, 469, 470, 471, 472, 173, 159, 33, 145, 263, 386, 398, 374, 308, 13, 78, 14]

# For webcam input:
cap = cv2.VideoCapture(0)
cv2.namedWindow("Test")
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    if results.multi_face_landmarks:
      points_mat = np.array([[int(results.multi_face_landmarks[0].landmark[i].x*image.shape[1]), int(results.multi_face_landmarks[0].landmark[i].y*image.shape[0])] for i in FACE])
      for point in points_mat:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
    cv2.imshow('Test', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
