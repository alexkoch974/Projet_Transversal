import cv2
import os
import numpy as np
import mediapipe as mp
from xml.dom import Node, minidom
from part import Moving_part
import time

EAR_L = [103, 109, 67]
EAR_R = [338, 332 , 297]
FACE = [152, 234, 10, 454]
EYE_L = [159, 145, 33, 173]
EYE_R =[386, 374, 398, 263]
MOUSTACHE_L = [187, 214]
MOUSTACHE_R = [411, 434]
MOUTH = [13, 14, 78, 308]

ALL_POINTS = EAR_L + EAR_R + FACE + EYE_L + EYE_R + MOUSTACHE_L + MOUSTACHE_R + MOUTH

keys = ['earL', 'earR', 'face', 'eyeL', 'eyeR', 'moustacheL', 'moustacheR', 'mouth']

paths = ['.\\avatar\\earL.png', 
         '.\\avatar\\earR.png', 
         '.\\avatar\\face.png',
         '.\\avatar\\eyeL.png',
         '.\\avatar\\eyeR.png',
         '.\\avatar\\moustacheL.png',
         '.\\avatar\\moustacheR.png',
         '.\\avatar\\mouth.png']
# paths = ['.\\avatar\\fuwa_glace.png']


def compute_translations(A : np.ndarray, B : np.ndarray) :
    return np.mean(A - B, 1)


def compute_tranformations(A : np.ndarray, B : np.ndarray) :
    '''
    Function that computes the transformations matrix from a human face to the avatar.
        
    Args :
        - A (np.ndarray) : points from the face in the current frame
        - B (np.ndarray) : points from the face in the calibration frame
        
    Retrun :
        - s : Squeeze vector [squeeze_on_x, squeeze_on_y]
        - theta : angle of the rotation
    '''
    M = B @ np.linalg.pinv(A)
    
    a = M[0,0]
    b = M[0,1]
    c = M[1,0]
    d = M[1,1]
    
    s = np.array([np.sqrt(a**2 + c**2), np.sqrt(b**2 + d**2)])
    if np.abs(np.arctan2(b, a)) > np.abs(np.arctan2(c, d)) :
        if np.arctan2(b, a) > np.pi/4 :
            theta = np.pi/4
        else :
            theta = np.arctan2(b, a)
    else :
        if np.arctan2(c, d) > np.pi/4 :
            theta = - np.pi/4
        else :
            theta = - np.arctan2(c, d)
        

   
    
    return s, theta, 




def calibrate_face(points_mat) :
    '''
    Function to get the points of interest on the input face from mediapipe process.
    
    Args : 
        - points_mat (list(np.ndarray)) : list of the points of interest on the face
    
    Return : 
        - dict : points of interest of the face
    '''
    dico = {
        'earL' : [], 
        'earR' : [], 
        'face' : [], 
        'eyeL' : [],
        'eyeR' : [],
        'moustacheL' : [],
        'moustacheR' : [],
        'mouth' : []
    }
    for i in range(len(points_mat)) :
        if i in range(3) :
            dico['earL'].append(np.array(points_mat[i]))
        elif i in range(3,6) :
            dico['earR'].append(np.array(points_mat[i]))
        elif i in range(6,10) : 
            dico['face'].append(np.array(points_mat[i]))
        elif i in range(10,14) :
            dico['eyeL'].append(np.array(points_mat[i]))
        elif i in range(14,18) : 
            dico['eyeR'].append(np.array(points_mat[i]))
        elif i in range(18,20) :
            dico['moustacheL'].append(np.array(points_mat[i]))
        elif i in range(20,22) :
            dico['moustacheR'].append(np.array(points_mat[i]))
        elif i in range(22,26) :
            dico['mouth'].append(np.array(points_mat[i]))
    
    for k in dico.keys() :
        dico[k] = np.array(dico[k])
            
    return dico


def mix_images(img_source: np.ndarray, img_overlay: np.ndarray, x: int | float, y: int | float) :
    """Overlay img_overlay on top of img_source at the position (x, y), using alpha
    channel of img_overlay.
    Args:
        img_source (np.ndarray): background image
        img_overlay (np.ndarray): overlay image
        x (int | float): x position
        y (int | float): y position
    Returns:
        np.ndarray: overlayed image
    """

    # Setup
    x, y = int(x), int(y)
    img = img_source.copy()

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = img_overlay[y1o:y2o, x1o:x2o, 3][:, :, np.newaxis] / 255.0
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    return img




def rotate_image(img: np.ndarray, angle: float, x: int | float, y: int | float) :
    """Rotate image.
    Args:
        img (np.ndarray): image
        angle (float): angle (in radians)
    Returns:
        np.ndarray: rotated image
    """

    height, width = img.shape[:2]
    image_center = (int(x), int(y))

    # Get rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, -angle*180/np.pi, 1.0)

    # Rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # Find the new width and height
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    # Rotate the image
    rotated_img = cv2.warpAffine(img, rotation_mat, (new_width, new_height))

    return rotated_img




def transform(points_face, calibration_face, avatar_pieces, result_image) :
    '''
    Function that computes and apply transformations on the differents parts of the avatar.
    
    Args :
        - points_face (dict) : points on the face for the current frame
        - calibration_face (dict) : points on the face for the calibration frame
        - avatar_pieces (list(np.ndarray)) : list of the images of the avatar parts
        - result_image (np.ndarray) : background image
        
    Return :
        - np.ndarray : the final image of the avatar to show for this frame
    '''
    # Changing the order to compute the face's transformation first and apply it to the other parts
    # local_keys = ['face', 'earL', 'earR', 'eyeL', 'eyeR', 'moustacheL', 'moustacheR', 'mouth']
    # temp = avatar_pieces[0].copy()
    # avatar_pieces[0] = avatar_pieces[1].copy()
    # avatar_pieces[1] = avatar_pieces[2].copy()
    # avatar_pieces[2] = temp
    
    for i in range(len(avatar_pieces)) :
        
        squeeze, theta = (1,0)
        t = np.array([0,0])

        
        ############################################################################################################################ - EARS
        
        if keys[i] == 'earL' or keys[i] == 'earR' :
            t = compute_translations(np.transpose(points_face[keys[i]]), np.transpose(calibration_face[keys[i]]))
            
        ############################################################################################################################ - FACE
        
        elif keys[i] == 'face' :
            squeeze, theta = compute_tranformations(np.transpose(points_face['face']), np.transpose(calibration_face['face']))
            center = calibration_face['face'][0]
            avatar_pieces[i] = rotate_image(avatar_pieces[i], theta, center[0], center[1])
            
        ############################################################################################################################ - EYES
        
        elif keys[i] == 'eyeL' or keys[i] == 'eyeR' :
            t = compute_translations(np.transpose(points_face[keys[i]]), np.transpose(calibration_face[keys[i]]))
            squeeze, _ = compute_tranformations(np.transpose(points_face[keys[i]]), np.transpose(calibration_face[keys[i]]))
            
        ############################################################################################################################ - MOUSTACHES
        
        elif keys[i] == 'moustacheL' or keys[i] == 'moustacheR' :
            t = compute_translations(np.transpose(points_face[keys[i]]), np.transpose(calibration_face[keys[i]]))
            
        ############################################################################################################################ - MOUTH
        
        elif keys[i] == 'mouth' :
            t = compute_translations(np.transpose(points_face['mouth']), np.transpose(calibration_face['mouth']))
            
        ############################################################################################################################
        
        result_image = mix_images(result_image, avatar_pieces[i], t[0], t[1])
    
    return result_image



def _main_() :
    
    mp_face_mesh = mp.solutions.face_mesh
    cv2.namedWindow("Preview")
    cv2.namedWindow('Result')
    calibration_face = {}
    not_calibrated = True
    calibration_done = False
    display_landmarks = True
    background_image = np.concatenate((cv2.imread('.\\avatar\\body.png', cv2.IMREAD_UNCHANGED), np.ones((480,480,1))*255.0), 2)
    homographies = []
    avatar_pieces_old = []
    for p in paths :
        avatar_pieces_old.append(cv2.imread(p, cv2.IMREAD_UNCHANGED))

    cap = cv2.VideoCapture(0) 
    detected_points_old = np.zeros((len(ALL_POINTS),2))
    detected_points = np.zeros((len(ALL_POINTS),2))
    
    with mp_face_mesh.FaceMesh(max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened(): 
            
            success, image = cap.read()
            image = image[:,80:560,:]
            result_image = background_image.copy()
            avatar_pieces = avatar_pieces_old.copy()
            homographies.clear()
            
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks is not None :
                results = results.multi_face_landmarks[0]
            else :
                continue

            
            detected_points = np.array([[int(results.landmark[i].x*image.shape[1]), int(results.landmark[i].y*image.shape[0])] for i in ALL_POINTS])
            distances = []
            for i in range(len(ALL_POINTS)) :
                distances.append(np.linalg.norm(detected_points[i,:] - detected_points_old[i,:]))
                
            if np.mean(np.array(distances)) < 2 :
                detected_points = detected_points_old.copy()
                
                
            if detected_points is not None :   
                if display_landmarks :            
                    for i in range(detected_points.shape[0]):
                        cv2.circle(image, (int(detected_points[i,0]), int(detected_points[i,1])), 1, (0, 255, 0), 1)
                    
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            cv2.putText(image, 'Try to reproduce the avatar\'s facial expression', (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, 'Then press Enter to start', (145,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, 'Press Esc or Q to quit', (150,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if display_landmarks : 
                cv2.putText(image, 'Press X to hide landmarks', (130,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else :
                cv2.putText(image, 'Press X to display landmarks', (130,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Preview', image)
                
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 :
                break
            if key == ord('x') :
                display_landmarks = not display_landmarks
            elif key == 13 :
                if not_calibrated :
                    not_calibrated = False
                    calibration_face = calibrate_face(list(detected_points))
                    calibration_done = True
                    display_landmarks = False
            
            if calibration_done :
                points_face = calibrate_face(list(detected_points))
                result_image = transform(points_face, calibration_face, avatar_pieces, result_image) 
            else :        
                for i in range(len(paths)) :    
                    result_image = mix_images(result_image, avatar_pieces[i], 0, 0)
                
            cv2.imshow('Result', cv2.flip(result_image/255.0, 1))
                
            detected_points_old = detected_points.copy()
            


            


    cap.release()
        
    

if __name__ == '__main__':
    _main_()
    
