import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

EAR_L = [103, 67, 109]
EAR_R = [338, 297, 332]
FACE = [152, 234, 10, 454]
EYE_L = [145, 476, 159, 474]
EYE_R = [374, 471, 386, 469]
MOUSTACHE_L = [214, 187]
MOUSTACHE_R = [434, 411]
MOUTH = [14, 78, 13, 308]

ALL_POINTS = EAR_L + EAR_R + FACE + EYE_L + EYE_R + MOUSTACHE_L + MOUSTACHE_R + MOUTH

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
    image = image[:,80:560,:]
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
      points_mat = np.array([[int(results.multi_face_landmarks[0].landmark[i].x*image.shape[1]), int(results.multi_face_landmarks[0].landmark[i].y*image.shape[0])] for i in ALL_POINTS])
      for point in points_mat:
        cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
    cv2.imshow('Test', cv2.flip(image, 1))
    key = cv2.waitKey(1)
    if key == ord('q'):
      break
cap.release()
