import cv2
import numpy as np

from utils import *



def click(event, x, y, flags, param):
    global saved_points, points

    closest = -1
    dist = -1
    for i in range(points.shape[0]):
        d = np.linalg.norm(points[i] - np.array([x, y]))
        if closest == -1 or d < dist:
            closest = i
            dist = d

    if event == cv2.EVENT_LBUTTONDOWN:
        saved_points.append(closest)
        print(saved_points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if closest in saved_points:
            saved_points.remove(closest)
        

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Test", click)

saved_points = []

point_detector = PointsDetector()
cap = cv2.VideoCapture(0)

prev_points = []

while True:

    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    points_ = point_detector.process(frame)

    if points_ is None:
        continue
    points = points_
    
    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, (255, 255, 255), cv2.FILLED)
    
    for point_id in saved_points:
        point = points[point_id]
        cv2.circle(frame, (int(point[0]), int(point[1])), 1, (0, 255, 0), cv2.FILLED)
        
    cv2.imshow("Test", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # q or ESC
        break