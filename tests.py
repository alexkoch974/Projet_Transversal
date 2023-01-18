import cv2
import os
import numpy as np
from main import mix_images


# images = ['earL', 'earR', 'body', 'face', 'crown', 'eyeL', 'eyeR', 'moustacheL', 'moustacheR', 'mouth']
# dico = {
#     'earL' : [], 
#     'earR' : [], 
#     'face' : [], 
#     'eyeL' : [],
#     'eyeR' : [],
#     'moustacheL' : [],
#     'moustacheR' : [],
#     'mouth' : []
# }
# saved_points = []
# img = None   

# cv2.namedWindow('Avatar')
 

# while True:
   
#     img = cv2.imread('.\\avatar\\earL.png', cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread('.\\avatar\\earR.png', cv2.IMREAD_UNCHANGED)
#     img3 = cv2.imread('.\\avatar\\face.png', cv2.IMREAD_UNCHANGED)
#     img = mix_images(img, img)
#     img = mix_images(img, img2)
#     img = mix_images(img, img3)
    
#     for i in range(len(saved_points)):
#         cv2.circle(img, (int(saved_points[i][0]), int(saved_points[i][1])), 1, (0, 0, 255), 3)
        

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
    
    
#     cv2.imshow("Avatar", img)
    
#     if len(dico['mouth']) == 4 :
#         break






im1 = cv2.imread('.\\avatar\\fuwa_glace.png', cv2.IMREAD_UNCHANGED)
print(im1[240,240,:])
im_bg = np.ones((480,480,4)) * 255.0
res = mix_images(im_bg, im1, 0, 0)
print(res[240,240,:])
cv2.namedWindow('zouz')
while True :
    cv2.imshow('zouz', res/255.0)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
            
