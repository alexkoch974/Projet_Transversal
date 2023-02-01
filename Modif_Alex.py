
import cv2
import mediapipe as mp
import numpy as np
from os.path import sep
import math



def get_delta_center(center, az, a):
    x = a* math.fabs(math.sin(az))
    y = a - a* math.fabs(math.cos(az))

    if az > 0:
        d = [center[0] + int(x) , center[1] + int(y) + 20]
    else:
        d = [center[0] - int(x) , center[1] +( int(y) + 20 ) ]
    

    return d
      


def translation(img, x, y):

    nb_l, nb_c, nb_ca = img.shape
    new_img = np.zeros((nb_l, nb_c, nb_ca))

    if x < 0:
        new_start_x = 0 #0 ou x (x > 0)
        new_end_x = nb_l -1 +x
        start_x = -x #0 ou x (x > 0)
        end_x = nb_l -1
    else:
        new_start_x = x #0 ou x (x > 0)
        new_end_x = nb_l -1
        start_x = 0 #0 ou x (x > 0)
        end_x = nb_l -1 -x
    if y < 0:
        new_start_y = 0 #0 ou x (x > 0)
        new_end_y = nb_l -1 +y
        start_y = -y #0 ou x (x > 0)
        end_y = nb_l -1
    else:
        new_start_y = y #0 ou x (x > 0)
        new_end_y = nb_l -1
        start_y = 0 #0 ou x (x > 0)
        end_y = nb_l -1 -y
    new_img[new_start_x:new_end_x, new_start_y:new_end_y, :] = img[start_x:end_x, start_y:end_y, :]
    
    return new_img

###types de mouvements
## gauche : petite rotazion du visage + translation !! une oreille passe devant
##> ou alors translation en z aussi 

def get_transformation(x, y, z, center, zp=0, zpp=200):

    #2d to 3d (projection)  , and -> rotation point - center point (origin point)
    proj2dto3d = np.array([[1,0,-center[0]],
                          [0,1,-center[1]],
                          [0,0,-zp],
                          [0,0,1]],np.float32)

    # 3d matrixs in  x ,y ,z 

    rx   = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]],np.float32)  #0

    ry   = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]],np.float32)

    rz   = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,0],
                     [0,0,0,1]],np.float32)  #0


    trans= np.array([[1,0,0,0],
                     [0,1,0,0],
                     [0,0,1,zpp],   #400 to move the image in z axis 
                     [0,0,0,1]],np.float32)




    proj3dto2d = np.array([ [200,0,center[0],0],
                            [0,200,center[1],0],
                            [0,0,1,0] ],np.float32)

    rx[1,1] = math.cos(x) #0
    rx[1,2] = -math.sin(x) #0
    rx[2,1] = math.sin(x) #0
    rx[2,2] = math.cos(x) #0
       
    ry[0,0] = math.cos(y)
    ry[0,2] = -math.sin(y)
    ry[2,0] = math.sin(y)
    ry[2,2] = math.cos(y)
        
    rz[0,0] = math.cos(z) #0
    rz[0,1] = -math.sin(z) #0
    rz[1,0] = math.sin(z) #0
    rz[1,1] = math.cos(z) #0
        
    r =rx.dot(ry).dot(rz) # if we remove the lines we put    r=ry
    M = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))

    return M

def segmentation(IMG):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)


    I = np.where(IMG[:,:,3] != 0)

    min_x = min(I[0])
    max_x = max(I[0])

    min_y = min(I[1])
    max_y = max(I[1])

##    print("min x = ", min_x)
##    print("max x = ", max_x)
##    print("min y = ", min_y)
##    print("max y = ", max_y)
##
##    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
##
##    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
##    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
##    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
##    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
##    cv2.imshow("all", image4)
    return (max_x, max_y), IMG[min_x:max_x+1, min_y:max_y+1, :]

def get_center(IMG):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)


    I = np.where(IMG[:,:,3] != 0)

    min_x = min(I[0])
    max_x = max(I[0])

    min_y = min(I[1])
    max_y = max(I[1])

##    print("min x = ", min_x)
##    print("max x = ", max_x)
##    print("min y = ", min_y)
##    print("max y = ", max_y)
##
##    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
##
##    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
##    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
##    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
##    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
##    cv2.imshow("all", image4)
    return [round( (min_x + max_x)/2), round( (min_y + max_y)/2)]
##def get_low_center(IMG):
##    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
##    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)
##
##
##    I = np.where(IMG[:,:,3] != 0)
##
##    min_x = min(I[0])
##    max_x = max(I[0])
##
##    min_y = min(I[1])
##    max_y = max(I[1])
##
####    print("min x = ", min_x)
####    print("max x = ", max_x)
####    print("min y = ", min_y)
####    print("max y = ", max_y)
####
####    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
####
####    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
####    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
####    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
####    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
####    cv2.imshow("all", image4)
##    return [max_x, round((min_y + max_y)/2)]
def get_low_center(IMG):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)


    I = np.where(IMG[:,:,3] != 0)

    min_x = min(I[0])
    max_x = max(I[0])

    min_y = min(I[1])
    max_y = max(I[1])

##    print("min x = ", min_x)
##    print("max x = ", max_x)
##    print("min y = ", min_y)
##    print("max y = ", max_y)
##
##    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
##
##    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
##    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
##    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
##    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
##    cv2.imshow("all", image4)
    #return [max_x, round((min_y + max_y)/2)]
    y = round((min_x + max_x)/2)
    x = min_y
    return [round((min_y + max_y)/2), max_x]


def get_right_center(IMG):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)


    I = np.where(IMG[:,:,3] != 0)

    min_x = min(I[0])
    max_x = max(I[0])

    min_y = min(I[1])
    max_y = max(I[1])

##    print("min x = ", min_x)
##    print("max x = ", max_x)
##    print("min y = ", min_y)
##    print("max y = ", max_y)
##
##    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
##
##    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
##    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
##    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
##    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
##    cv2.imshow("all", image4)
    #return [max_x, round((min_y + max_y)/2)]
    y = round((min_x + max_x)/2)
    x = min_y
    return [max_y, round((min_x + max_x)/2)]

def get_left_center(IMG):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)


    I = np.where(IMG[:,:,3] != 0)

    min_x = min(I[0])
    max_x = max(I[0])

    min_y = min(I[1])
    max_y = max(I[1])

##    print("min x = ", min_x)
##    print("max x = ", max_x)
##    print("min y = ", min_y)
##    print("max y = ", max_y)
##
##    cv2.imshow("eye", IMG[min_x:max_x+1, min_y:max_y+1, :])
##
##    image = cv2.circle(IMG.copy(), (min_y,min_x), radius=1, color=(0, 255, 255))
##    image2 = cv2.circle(image, (min_y,max_x), radius=1, color=(0, 255, 255))
##    image3 = cv2.circle(image2, (max_y,min_x), radius=1, color=(0, 255, 255))
##    image4 = cv2.circle(image3, (max_y,max_x), radius=1, color=(0, 255, 255))
##    cv2.imshow("all", image4)
    #return [max_x, round((min_y + max_y)/2)]
    y = round((min_x + max_x)/2)
    x = min_y
    return [min_y, round((min_x + max_x)/2)]



def glue(bounds, crop, s):
    #PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
    #IMG = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)
    h, w = s #IMG.shape[:2]
    h_c, w_c = crop.shape[:2]

    IMG = np.zeros((h, w, 4))

    IMG[bounds[0]-h_c+1:bounds[0]+1, bounds[1]-w_c+1:bounds[1]+1, :] = crop 

    return IMG

def add_image_using_alpha(img1, img2):
    """Add im2 on im1 where img[:, :, alpha] != 0"""

##    face = cv2.imread('./avatar/face.png', cv2.IMREAD_UNCHANGED)
##    cv2.imshow('face', face)
##    mouth = cv2.imread('./avatar/mouth.png', cv2.IMREAD_UNCHANGED)
##    cv2.imshow('mouth', mouth)


    nb_li, nb_co, nb_ca = img1.shape

    img3 = img1

    alpha_not_null = np.where(img2[:,:,3] != 0)

    img3[alpha_not_null[0], alpha_not_null[1], :] = img2[alpha_not_null[0], alpha_not_null[1], :]

    return img3


LEFT_IRIS = [473, 474, 475, 476, 477]
NP_LEFT_IRIS = np.zeros((len(LEFT_IRIS),2))
  
RIGHT_IRIS = [468, 469, 470, 471, 472]
NP_RIGHT_IRIS = np.zeros((len(RIGHT_IRIS),2))

LEFT_EYE = [173, 159, 33, 145]
NP_LEFT_EYE = np.zeros((len(LEFT_EYE),2))

RIGHT_EYE = [263, 386, 398, 374]
NP_RIGHT_EYE = np.zeros((len(RIGHT_EYE),2))

MOUTH = [308, 13, 78, 14]
NP_MOUTH = np.zeros((len(MOUTH),2))

FACE_PATH = "D:/User/s9/transversal/avatar/face.png"
FACE_IMG = cv2.imread(FACE_PATH, cv2.IMREAD_UNCHANGED)
BODY_PATH = "D:/User/s9/transversal/avatar/body.png"
BODY_IMG = cv2.imread(BODY_PATH, cv2.IMREAD_UNCHANGED)
EYEL_PATH = "D:/User/s9/transversal/avatar/background_eyeL.png"
EYEL_IMG = cv2.imread(EYEL_PATH, cv2.IMREAD_UNCHANGED)
B_EYEL, CROP_EYEL = segmentation(EYEL_IMG)
EYER_PATH = "D:/User/s9/transversal/avatar/background_eyeR.png"
EYER_IMG = cv2.imread(EYER_PATH, cv2.IMREAD_UNCHANGED)
B_EYER, CROP_EYER = segmentation(EYER_IMG)



EARL_PATH = "D:/User/s9/transversal/avatar/earL.png"
EARL_IMG = cv2.imread(EARL_PATH, cv2.IMREAD_UNCHANGED)
EARR_PATH = "D:/User/s9/transversal/avatar/earR.png"
EARR_IMG = cv2.imread(EARR_PATH, cv2.IMREAD_UNCHANGED)


MR_PATH = "D:/User/s9/transversal/avatar/background_moustacheR.png"
MR_IMG = cv2.imread(MR_PATH, cv2.IMREAD_UNCHANGED)
ML_PATH = "D:/User/s9/transversal/avatar/background_moustacheL.png"
ML_IMG = cv2.imread(ML_PATH, cv2.IMREAD_UNCHANGED)

CROWN_PATH = "D:/User/s9/transversal/avatar/crown.png"
CROWN_IMG = cv2.imread(CROWN_PATH, cv2.IMREAD_UNCHANGED)

MOUTH_PATH = "D:/User/s9/transversal/avatar/background_mouth.png"
MOUTH_IMG = cv2.imread(MOUTH_PATH, cv2.IMREAD_UNCHANGED)
B_MOUTH, CROP_MOUTH = segmentation(MOUTH_IMG)

HAUTEUR, LARGEUR = FACE_IMG.shape[:2]

center = get_low_center(FACE_IMG)#[236, 356] #menton
center_face = get_center(FACE_IMG)#[236, 250] #centre de la tête
print(center_face)

center_earL = get_low_center(EARL_IMG)# (192, 215) # bas milieu oreille g
center_earR = get_low_center(EARR_IMG)#(279, 215)# bas milieu oreille d
center_ear = [round((center_earL[0] + center_earL[0])/2), round((center_earL[1] + center_earL[1])/2)]#(245, 215) # bas milieu des 2 oreilles
center_crown = get_low_center(CROWN_IMG)#(236, 222) # bas milieu oreille crown
center_moustL = get_right_center(ML_IMG)# (175, 310)
center_moustR = get_left_center(MR_IMG)# (296, 310)

##
##
###2d to 3d (projection)  , and -> rotation point - center point (origin point)
##proj2dto3d = np.array([[1,0,-center[0]],
##                      [0,1,-center[1]],
##                      [0,0,0],
##                      [0,0,1]],np.float32)
##
### 3d matrixs in  x ,y ,z 
##
##rx   = np.array([[1,0,0,0],
##                 [0,1,0,0],
##                 [0,0,1,0],
##                 [0,0,0,1]],np.float32)  #0
##
##ry   = np.array([[1,0,0,0],
##                 [0,1,0,0],
##                 [0,0,1,0],
##                 [0,0,0,1]],np.float32)
##
##rz   = np.array([[1,0,0,0],
##                 [0,1,0,0],
##                 [0,0,1,0],
##                 [0,0,0,1]],np.float32)  #0
##
##
##trans= np.array([[1,0,0,0],
##                 [0,1,0,0],
##                 [0,0,1,200],   #400 to move the image in z axis 
##                 [0,0,0,1]],np.float32)
##
##
##
##
##proj3dto2d = np.array([ [200,0,center[0],0],
##                        [0,200,center[1],0],
##                        [0,0,1,0] ],np.float32)




ax_moust = 0
sens_moust = 1
az_ear = 0
sens_ear = 1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

ANGLES_SIZE = 2
angles = np.zeros((3, ANGLES_SIZE))
i_angles = 0
last_angles = np.array([_ for _ in range(ANGLES_SIZE)], dtype=int)

ax = 0
ay = 0
az = 0



with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:



    while cap.isOpened():
      success, image = cap.read()
      nb_l = image.shape[0]
      nb_c = image.shape[1]

    
      if not success:
        print("Ignoring empty camera frame.")
      
    

    # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      results = face_mesh.process(image)
      if results.multi_face_landmarks:
          points = results.multi_face_landmarks[0].landmark
          for k in range(len(points)):
              x = round(points[k].x * nb_c)
              y = round(points[k].y * nb_l)
##              if k in [10, 152, 234, 454]:
              if k in [234]:
                  image = cv2.circle(image, (x,y), radius=1, color=(0, 255, 255), thickness=1)
              else:
                  image = cv2.circle(image, (x,y), radius=1, color=(0, 0, 255), thickness=1)
            
        
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))


      #### FACE
      height, width = FACE_IMG.shape[:2]


      p1z = np.array([points[10].x, points[10].y])
      p2z = np.array([points[152].x, points[152].y])
      vz = (p2z - p1z) / np.linalg.norm(p2z - p1z)
      cos_z = vz[0]

      p1y = np.array([points[234].x, points[234].z])
      p2y = np.array([points[454].x, points[454].z])
      vy = - (p2y - p1y) / np.linalg.norm(p2y - p1y)
      cos_y = vy[1]

      p1x = np.array([points[1].y, points[1].z])
      p2x = np.array([ (points[234].y + points[454].y) / 2, (points[234].z + points[454].z) / 2])
      vx = - (p2x - p1x) / np.linalg.norm(p2x - p1x)
      cos_x = vx[0]


      ax_ = - math.acos(cos_x) + math.pi / 2
      ay_ = - math.acos(cos_y) + math.pi / 2
      az_ = - math.acos(cos_z) + math.pi / 2

      angles[0, i_angles] = ax_
      angles[1, i_angles] = ay_
      angles[2, i_angles] = az_
      i_angles = (i_angles +1) % ANGLES_SIZE

      
      ax -= (angles[0, i_angles] - angles[0, (i_angles -ANGLES_SIZE +1)%ANGLES_SIZE]) / ANGLES_SIZE
      ay -= (angles[1, i_angles] - angles[1, (i_angles -ANGLES_SIZE +1)%ANGLES_SIZE]) / ANGLES_SIZE
      az -= (angles[2, i_angles] - angles[2, (i_angles -ANGLES_SIZE +1)%ANGLES_SIZE]) / ANGLES_SIZE
      
##      ax = ax_
##      ay = ay_
##      az = az_


      ay_face = max(min(ay / 2, 0.10),-0.10)




      if sens_moust > 0:


          if ax_moust < 0.3:
              ax_moust += 0.02
          else :
              sens_moust = -1
              ax_moust -= 0.02
      else :
          if ax_moust > -0.1:
              ax_moust -= 0.02
          else :
              sens_moust = 1
              ax_moust += 0.02

      if sens_ear > 0:


          if az_ear < 0.15:
              az_ear += 0.01
          else :
              sens_ear = -1
              az_ear -= 0.01
      else :
          if az_ear > 0:
              az_ear -= 0.01
          else :
              sens_ear = 1
              az_ear += 0.01

      a = center[1] - center_face[1]
      print(ax)
      d_center_face = get_delta_center(center_face, az, a)
      
      d_center_ear = get_delta_center(center_ear, az, center[1] - 215)
      #d_center_earL = get_delta_center((192, 215), az, center[1] - 215)
      #d_center_earR = get_delta_center((279, 215), az, a)
      d_center_crown = get_delta_center(center_crown, az, center[1] -  222)
      
          
      

      
      M_face = get_transformation(ax, ay, az, d_center_face, zp=80, zpp=300)

      M_ear = get_transformation(ax, ay, az, d_center_face, zp=-10)
      M_earL = get_transformation(0, 0, az_ear, center_earL, zpp=200)
      M_earR = get_transformation(0, 0, -az_ear, center_earR, zpp=200)

      M_face_only = get_transformation(0, ay_face, az, center)
      M_crown = get_transformation(min(0.4, ax), 0, 0, (236, 222))
      M_crown2 = get_transformation(0, 0, az, center)

      
      M_moust = get_transformation(ax, ay, 0, center, zp=80, zpp=300)
      M_moust2 = get_transformation(0, 0, az, d_center_face)

      
      M_moustL = get_transformation(0, 0, ax_moust, center_moustL)
      M_moustR = get_transformation(0, 0, -ax_moust, center_moustR)
      
##      rx[1,1] = math.cos(ax) #0
##      rx[1,2] = -math.sin(ax) #0
##      rx[2,1] = math.sin(ax) #0
##      rx[2,2] = math.cos(ax) #0
##        
##      ry[0,0] = math.cos(ay)
##      ry[0,2] = -math.sin(ay)
##      ry[2,0] = math.sin(ay)
##      ry[2,2] = math.cos(ay)
##        
##      rz[0,0] = math.cos(az) #0
##      rz[0,1] = -math.sin(az) #0
##      rz[1,0] = math.sin(az) #0
##      rz[1,1] = math.cos(az) #0
##        
##      r =rx.dot(ry).dot(rz) # if we remove the lines we put    r=ry
##      final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))


        
        
      dst = cv2.warpPerspective(FACE_IMG.copy(), M_face_only,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      #dst = FACE_IMG.copy()
      

     ### EYES

##      m1 = (points[LEFT_IRIS[0]].x - points[LEFT_EYE[-2]].x) - (points[LEFT_IRIS[0]].x - points[LEFT_IRIS[-2]].x)
      m1 = (points[RIGHT_IRIS[-2]].x - points[RIGHT_EYE[-2]].x)
      x1 = (round(m1 * nb_l ) -70) * 3
      M1 = np.float32([[1,0,-x1],[0,1,0]])
      dst_eyeL = cv2.warpAffine(EYEL_IMG.copy(),M1,(nb_c,nb_l))

      ##if (points[LEFT_IRIS[2]].y - points[LEFT_IRIS[4]].y) - (points[LEFT_EYE[1]].y - points[LEFT_EYE[3]].y) :
      eye_ratio = max(0.01, min((points[RIGHT_EYE[1]].y - points[RIGHT_EYE[3]].y) / (points[RIGHT_IRIS[2]].y - points[RIGHT_IRIS[4]].y), 1) )

      #b_eyeL, crop_eyeL = segmentation(dst_eyeL)
      cv2.imshow("1", CROP_EYEL)
      w_c, h_c = CROP_EYEL.shape[:2]
      #print(w_c, h_c)
      dst_eyeL2 = cv2.resize(CROP_EYEL, (h_c, round(eye_ratio * w_c)), interpolation = cv2.INTER_AREA)
      cv2.imshow("2", dst_eyeL2)
      dst_eyeL2 = glue(B_EYEL, dst_eyeL2, (HAUTEUR, LARGEUR))
      dst_eyeL3 = cv2.warpPerspective(dst_eyeL2, M_face,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR
                                  ,cv2.BORDER_CONSTANT,(255,255,255))
      cv2.imshow("eye3",dst_eyeL3)
      #dst_eyeL2 = cv2.resize(dst_eyeL, (480, round(eye_ratio * 480)), interpolation = cv2.INTER_AREA)
      ##else:
          ##dst_eyeL2 = dst_eyeL

##      x2 = round((points[RIGHT_IRIS[-2]].x - points[RIGHT_EYE[-2]].x + 0.5) * nb_l)
##      M2 = np.float32([[1,0,-x1],[0,1,0]])
##      dst_eyeR = cv2.warpAffine(EYER_IMG.copy(),M2,(nb_c,nb_l))


      m2 = (points[LEFT_IRIS[-2]].x - points[LEFT_EYE[-2]].x)
      x2 = x1#(round(m2 * nb_l ) -70) * 3
      M2 = np.float32([[1,0,-x2],[0,1,0]])
      dst_eyeR = cv2.warpAffine(EYER_IMG.copy(),M2,(nb_c,nb_l))
      cv2.imshow("3b", dst_eyeR)
      #print(x2)

      eye_ratio2 = max(0.01, min((points[LEFT_EYE[1]].y - points[LEFT_EYE[3]].y) / (points[LEFT_IRIS[2]].y - points[LEFT_IRIS[4]].y), 1) )

      #b_eyeR, crop_eyeR = segmentation(dst_eyeR)
      cv2.imshow("3", CROP_EYER)
      w_c, h_c = CROP_EYER.shape[:2]
      dst_eyeR2 = cv2.resize(CROP_EYER, (h_c, round(eye_ratio2 * w_c)), interpolation = cv2.INTER_AREA)
      cv2.imshow("4", dst_eyeR2)
      dst_eyeR2 = glue(B_EYER, dst_eyeR2, (HAUTEUR, LARGEUR))
      dst_eyeR3 = cv2.warpPerspective(dst_eyeR2, M_face,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR
                                  ,cv2.BORDER_CONSTANT,(255,255,255))


      #### MOUTH

      mouth_ratio =  points[MOUTH[3]].y - points[MOUTH[1]].y#max(eye_ratio, 0.1) #max(0.01, min((points[LEFT_EYE[1]].y - points[LEFT_EYE[3]].y) / (points[LEFT_IRIS[2]].y - points[LEFT_IRIS[4]].y), 1) )
      #print(mouth_ratio*10)
      mouth_ratio = max(0.1, mouth_ratio*10)
      
      w_c, h_c = CROP_MOUTH.shape[:2]
      dst_mouth = cv2.resize(CROP_MOUTH, (h_c, round(mouth_ratio * w_c)), interpolation = cv2.INTER_AREA)
      cv2.imshow("2", dst_mouth)
      dst_mouth2 = glue(B_MOUTH, dst_mouth, (HAUTEUR, LARGEUR))
      dst_mouth3 = cv2.warpPerspective(dst_mouth2, M_face,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR
                                  ,cv2.BORDER_CONSTANT,(255,255,255))


      ### MOUSTACHE R

      dst_moustR = cv2.warpPerspective(MR_IMG.copy(), M_moust,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_moustL = cv2.warpPerspective(ML_IMG.copy(), M_moust,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_moustR1 = cv2.warpPerspective(dst_moustR, M_moust2,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_moustL1 = cv2.warpPerspective(dst_moustL, M_moust2,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_moustR2 = cv2.warpPerspective(dst_moustR1, M_moustR,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_moustL2 = cv2.warpPerspective(dst_moustL1, M_moustL,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      
      dst_moust = add_image_using_alpha(dst_moustL2, dst_moustR2)

      ### EAR

      dst_earL = cv2.warpPerspective(EARL_IMG.copy(), M_ear,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_earR = cv2.warpPerspective(EARR_IMG.copy(), M_ear,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_earL2 = cv2.warpPerspective(dst_earL, M_earL,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_earR2 = cv2.warpPerspective(dst_earR, M_earR,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      

    
      if ay < -0.2 :

          back_ear = dst_earL2
          front_ear = dst_earR2
      elif ay > 0.2 :
          back_ear = dst_earR2
          front_ear = dst_earL2

      else :

          if ax < 0.3:

              back_ear = add_image_using_alpha(dst_earL2, dst_earR2)
              front_ear = np.zeros((HAUTEUR, LARGEUR, 4))

          else :

              back_ear = np.zeros((HAUTEUR, LARGEUR, 4))
              front_ear = add_image_using_alpha(dst_earL2, dst_earR2)
              


      ### CROWN

      

      x_crown = round(ay*50)
      y_crown = max(0, round(ax*100))
          

      
      dst_crown = translation(CROWN_IMG.copy(), -10 + y_crown, x_crown)
      #print(ax, y_crown)
      dst_crown2 = cv2.warpPerspective(dst_crown, M_crown,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      dst_crown3 = cv2.warpPerspective(dst_crown2, M_crown2,(FACE_IMG.shape[1],FACE_IMG.shape[0]),None,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,(255,255,255))
      
      
           

      dst_b = add_image_using_alpha(back_ear, dst)
      dst_b2 = add_image_using_alpha(dst_b, front_ear)
      
          
      dst2 = add_image_using_alpha(BODY_IMG.copy(), dst_b2)
      dst2b = add_image_using_alpha(dst2, dst_mouth3)

      dst_eyes = add_image_using_alpha(dst_eyeL3, dst_eyeR3)
      dst3 = add_image_using_alpha(dst2b, dst_eyes)

      #dst_moust2 = translation(dst_moust, -10, 0)
      dst4 = add_image_using_alpha(dst3, dst_moust)

      dst_5 = add_image_using_alpha(dst4, dst_crown3)

      test = cv2.circle(dst_5.copy(), (d_center_crown[0], d_center_crown[1]), radius=1, color=(0, 0, 255), thickness=1)
      test2 = cv2.circle(test, center_face, radius=1, color=(0, 0, 255), thickness=1)
      test3 = cv2.circle(test2, (d_center_ear[0], d_center_ear[1]), radius=1, color=(255, 255, 0), thickness=1)
     
      cv2.imshow("test3", test3)

      cv2.imshow("avatar2", dst_5)
      #cv2.imshow("avatar1", dst_eyeL2)


      if cv2.waitKey(5) & 0xFF == 27:
        break
      

      
cap.release()




#warpPerspective
#findHomography



### recuperer le menton pour la calibration
