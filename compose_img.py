import os
import cv2
import numpy as np


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


# list files data/avatar directory
files = ['body.png', 'earL.png', 'earR.png', 'head.png', 'crown.png', 'eyeL.png', 'eyeR.png', 'mouth.png', 'moustacheL.png', 'moustacheR.png']
positions = [[587, 1274], [353, 180], [971, 181], [474, 699], [746, 594], [773, 900], [930, 899], [818, 1182], [70, 1017], [1099, 1011]]

# load ref image
ref_img = cv2.imread('data/avatar/fuwa_glace.png', cv2.IMREAD_UNCHANGED)
mse = -1

# create window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

test_id = 0
scale = 1

running = len(files)
while running > 0:

    previus_mse = mse
    move = (0, 0)

    for tx, ty in [(scale, 0), (-scale, 0), (0, scale), (0, -scale)]:

        img = np.zeros((1800, 1800, 4), np.uint8)
        for i, pos, file in zip(range(len(files)), positions, files):
            # load image
            img_path = os.path.join('data', 'avatar', file)
            img_ = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            # overlay image
            img = overlay_image_alpha(img, img_, pos[0] + (tx if i == test_id else 0), pos[1] + (ty if i == test_id else 0))

        # compute mse
        mse_ = np.mean((ref_img - img) ** 2)
        if mse_ < mse or mse == -1:
            mse = mse_
            move = (tx, ty)

    positions[test_id][0] += move[0]
    positions[test_id][1] += move[1]

    img = np.zeros((1800, 1800, 4), np.uint8)
    # draw all files
    for file, pos in zip(files, positions):
        # read image
        img_ = cv2.imread(os.path.join('data/avatar', file), cv2.IMREAD_UNCHANGED)
        # draw image
        img = overlay_image_alpha(img, img_, pos[0], pos[1])

    # compute mse
    mse = ((img - ref_img)**2).mean(axis=None)

    if previus_mse == mse:
        running -= 1
    else:
        running = len(files)

    test_id += 1
    test_id %= len(files)

    # show image
    cv2.imshow('image', img)

    # wait for key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    print(f"Pos: {positions}\nMSE: {mse}")