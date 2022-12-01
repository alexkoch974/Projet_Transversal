import cv2
import numpy as np

from utils import *


LEFT = 93
RIGHT = 323
NOSE = 4

point_detector = PointsDetector()
cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

running = True
while running:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Detect points
    points = point_detector.process(frame)
    if points is None:
        continue

    origin_point = (points[LEFT] + points[RIGHT]) / 2
    axe = points[NOSE] - origin_point
    axe = axe / np.linalg.norm(axe)

    # Project points on axe plane
    points = points - origin_point
    points = points - np.dot(points, axe)[:, np.newaxis] * axe

    # Project points on frame plane
    points = points + origin_point

    # Draw points
    for point in points[:, :2]:
        cv2.circle(frame, tuple(point.astype(int)), 1, (255, 255, 255), -1)
    
    # Draw main points
    for point in [LEFT, RIGHT, NOSE]:
        cv2.circle(frame, tuple(points[point, :2].astype(int)), 1, (0, 0, 255), -1)
    
    # Draw origin point
    cv2.circle(frame, tuple(origin_point[:2].astype(int)), 1, (255, 0, 0), -1)

    # Draw axes
    cv2.line(frame, tuple(origin_point[:2].astype(int)), tuple((origin_point[:2] + 20 * axe[:2]).astype(int)), (0, 0, 255), 1)

    # Show image
    cv2.imshow('Frame', frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        running = False
