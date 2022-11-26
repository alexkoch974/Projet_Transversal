import os
import cv2
import numpy as np

points = []

def click(event, x, y, flags, param):
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        point = [x - img.shape[1]/2, y - img.shape[0]/2]
        points.append(point)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()


def overlay_image_alpha(img_source, img_overlay, x, y):
    img = img_source.copy()

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = img_overlay[y1o:y2o, x1o:x2o, 3][:, :, np.newaxis] / 255.0
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    return img



img_path = os.path.join('imgs', 'avatar', 'mouth.png')
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

cv2.namedWindow("Test")
cv2.setMouseCallback("Test", click)

while True:
    img_ = np.ones_like(img) * 255
    img_ = overlay_image_alpha(img_, img, 0, 0)
    for point in points:
        cv2.circle(img_, (int(point[0] + img.shape[1]/2), int(point[1] + img.shape[0]/2)), 2, (0, 0, 255), cv2.FILLED)
    cv2.imshow("Test", img_)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break
    elif key == 13:
        print(points)
        break
