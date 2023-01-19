import cv2
import mediapipe as mp
import numpy as np
import time

# Open avatar all_forward.png
avatar = cv2.imread('data/avatar/all_forward.png')

# Image segmentation of avatar features with active contour

# Convert avatar to grayscale
avatar_gray = cv2.cvtColor(avatar, cv2.COLOR_BGR2GRAY)

# Threshold avatar
ret, thresh = cv2.threshold(avatar_gray, 127, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(avatar, contours, -1, (0, 255, 0), 3)

# Show image
cv2.imshow('Avatar', avatar)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()



