import cv2
import mediapipe as mp
import time
import numpy as np


cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

saved_points = []
scale = 2
interest_points = [10, 152, 234, 454, 473, 474, 475, 476, 477, 468, 469, 470, 471, 472, 173, 159, 33, 145, 263, 386, 398, 374, 308, 13, 78, 14]

def click(event, x, y, flags, param):
    global saved_points, points_mat, scale

    closest = -1
    dist = -1
    for i in range(points_mat.shape[0]):
        d = np.linalg.norm(points_mat[i] - np.array([x/scale, y/scale]))
        if closest == -1 or d < dist:
            closest = i
            dist = d

    if event == cv2.EVENT_LBUTTONDOWN:
        saved_points.append(closest)
        print(saved_points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if closest in saved_points:
            saved_points.remove(closest)
        

cv2.namedWindow("Test")
cv2.setMouseCallback("Test", click)

while True:
    success, img = cap.read()
    if not success:
        continue
    result = faceMesh.process(img)
    if result.multi_face_landmarks:
        points = result.multi_face_landmarks[0].landmark
        points_mat = np.array([[int(point.x*img.shape[1]), int(point.y*img.shape[0])] for point in points])
        
        for point in points_mat:
            cv2.circle(img, (point[0], point[1]), 2, (255, 255, 255), cv2.FILLED)
        
        for point_id in interest_points:
            point = points_mat[point_id]
            cv2.circle(img, (point[0], point[1]), 2, (0, 0, 255), cv2.FILLED)
        
        for point_id in saved_points:
            point = points_mat[point_id]
            cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), cv2.FILLED)
        

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Test", cv2.resize(img, (0,0), fx=scale, fy=scale))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break