import cv2
import numpy as np

def get_baricentric(p, v1, v2, v3, v4):
    a = v1 - p
    b = v2 - v1
    c = v4 - v1
    d = v1 - v2 - v4 + v3
    # we use (x,y) instead of (lambda, mu) for easy code
    # solution is supposed to have 0 <= x <=1 and 0 <= y <=1 
    # initial point
    x = 0.5
    y = 0.5
    dx = 0.1 * np.ones((2,1))
    iter = 0
    tol = 1e-12
    while np.linalg.norm(dx) > tol and iter < 20:
        # apply Newton-Rapson method to solve f(x,y)=0
        f = a + b*x + c*y + d*x*y
        # Newton: x_{n+1} = x_n - (Df^-1)*f
        # or equivalently denoting dx = x_{n+1}-x_n
        # Newton: Df*dx=-f
        Df = np.zeros((2,2))
        Df[:,0] = b + d*y  # df/dx
        Df[:,1] = c + d*x  # df/dy
        bb = -f.T # independent term
        dx = np.linalg.pinv(Df) @ bb
        x = x + dx[0]
        y = y + dx[1]
        iter = iter + 1
        if np.linalg.norm([x, y]) > 10:
            iter=20 # non convergent: just to save time

    if iter < 20:
        lamb=x
        mu=y
        alpha1=(1-mu)*(1-lamb)
        alpha2=lamb*(1-mu)
        alpha3=mu*lamb
        alpha4=(1-lamb)*mu
        alphas=[alpha1,alpha2,alpha3,alpha4]
    else:
        alphas=[-1,-1,-1,-1]; # wrong values
    return alphas

ref_points = []
points = []

def click_handler_refpt(event, x, y, flags, param):
    global ref_points

    if event == cv2.EVENT_LBUTTONDOWN:
        print('x: {}, y: {}'.format(x, y))
        ref_points.append([x, y])

def click_handler_pt(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        print('x: {}, y: {}'.format(x, y))
        points.append([x, y])

img = cv2.imread('data/avatar/mouth.png', cv2.IMREAD_UNCHANGED)
res = np.ones((1000, 1000, 4), dtype=np.uint8) * 255

# get reference points on img using click handler
cv2.namedWindow('image')
cv2.setMouseCallback('image', click_handler_refpt)
cv2.imshow('image', img)
while len(ref_points) < 4:
    cv2.waitKey(100)
cv2.destroyAllWindows()
ref_points = np.array(ref_points)

# get points on res clicking on the image
points = []
cv2.namedWindow('res')
cv2.setMouseCallback('res', click_handler_pt)
cv2.imshow('res', res)
while len(points) < 4:
    cv2.waitKey(100)
cv2.destroyAllWindows()
points = np.array(points)


# get baricentric coordinates of corners
corners = np.array([
    [1, 1],
    [img.shape[1], 0],
    [img.shape[1], img.shape[0]],
    [0, img.shape[0]]
])
baricentric_corners = np.array([
    get_baricentric(corner, ref_points[0], ref_points[1], ref_points[2], ref_points[3])
    for corner in corners
])

# compute coorners position in the res image
corners_res = (points.T @ baricentric_corners.T).T

# # draw img using the corners
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         baricentric = get_baricentric(np.array([j, i]), corners[0], corners[1], corners[2], corners[3])
#         if np.min(baricentric) < 0:
#             continue
#         pos = corners_res.T @ baricentric
#         color = img[i, j]
#         cv2.circle(res, tuple(pos.astype(np.int32)), 1, (int(color[0]), int(color[1]), int(color[2]), int(color[3])), -1)

for point in points:
    cv2.circle(res, tuple(point), 2, (0, 0, 255, 255), -1)

for corner in corners_res:
    cv2.circle(res, tuple(corner.astype(np.int32)), 2, (255, 0, 0, 255), -1)

cv2.imshow('res', res)
cv2.waitKey(0)