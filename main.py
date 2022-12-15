import cv2
import os
import numpy as np
import mediapipe as mp
from xml.dom import Node, minidom
from part import Moving_part
import time


EAR_L = [103, 109, 67]
EAR_R = [338, 332 , 297]
FACE = [10, 152, 234, 454]
EYE_L = [159, 145, 33, 173]
EYE_R =[386, 374, 398, 263]
MOUSTACHE_L = [187, 214]
MOUSTACHE_R = [411, 434]
MOUTH = [13, 14, 78, 308]

ALL_POINTS = EAR_L + EAR_R + FACE + EYE_L + EYE_R + MOUSTACHE_L + MOUSTACHE_R + MOUTH

paths = [os.path('.', 'avatar', 'earL.png'), 
         os.path('.', 'avatar', 'earR.png'), 
         os.path('.', 'avatar', 'face.png'),
         os.path('.', 'avatar', 'eyeL.png'),
         os.path('.', 'avatar', 'eyeR.png'),
         os.path('.', 'avatar', 'moustacheL.png'),
         os.path('.', 'avatar', 'moustacheR.png'),
         os.path('.', 'avatar', 'mouth.png')]


def get_avatar_points(calibration_file = 'calibration_avatar.xml'):
    '''
    Function that parses the calibration file of the avatar to get the points of interest.
    
    Args : 
        - calibration_file (str) : the path of the xml calibration file 
    
    Return : 
        - dict (keys : Moving_part) : points of interest of the avatar
    '''
    # Initialize dictionary
    dico = {}
    
    # Parser le fichier avec minidom
    doc = minidom.parse(calibration_file)
    root = doc.getElementsByTagName(str(doc.firstChild.tagName))
    avatar = root.item(0)
    for moving_part in avatar.childNodes :
        temp_list = []
        temp_part = None
        if moving_part.nodeType != Node.TEXT_NODE :
            for point in moving_part.childNodes :
                if point.nodeType != Node.TEXT_NODE :
                    if point.hasAttribute('x') :
                        temp_list.append(np.array([int(point.attributes.item(0).value), int(point.attributes.item(1).value), 1]))
                    else :
                        temp_list.append(None)
            temp_part = Moving_part(temp_list)            
            new_key = {moving_part.tagName : temp_part}
            dico.update(new_key)    
            
    return dico


def compute_calibration_transformations(calibration_avatar, calibration_face) :
    
    # for key in calibration_face.keys() :
        key = 'mouth'
        face_list = []
        for points in calibration_face[key] :
            face_list.append(points)
        input_face = np.array(face_list)          
        input_avatar = np.array(calibration_avatar[key].get_points())
        if input_face == [] or input_avatar == [] :
            H = None
            flag = False
        elif input_face.shape[0] < 4 or input_avatar.shape[0] < 4 :
            H = None
            flag = False
        else :   
            H, _ = cv2.findHomography(input_avatar,input_face)
            flag = True
    
        return H, flag


def calibrate_face(points_mat) :
    '''
    Function to get the points of interest on the input face from mediapipe process.
    
    Args : 
        - points_mat (list(np.array)) : list of the points of interest on the face
    
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
        if len(points_mat[i]) == 2 :
            points_mat[i].append(1)
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
            
    
    return dico


def _main_() :
    
    mp_face_mesh = mp.solutions.face_mesh
    calibration_avatar = get_avatar_points('.\\fuwa_glace_calibration.xml')
    calibration_face = {}
    not_calibrated = True
    avatar_ready = False
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Face Calibration")
    cv2.namedWindow("Preview")
    
    avatar_pieces = []
    for p in paths :
        avatar_pieces.append(cv2.imread(p))    
    
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            
            success, image = cap.read()
            preview_image = image.copy()
            image = image[:,80:560,:]
            
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            points_mat = []
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                points_mat = [[int(results.multi_face_landmarks[0].landmark[i].x*image.shape[1]),
                                        int(results.multi_face_landmarks[0].landmark[i].y*image.shape[0])]
                                        for i in ALL_POINTS]
                for point in points_mat:
                    cv2.circle(image, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
                    
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            cv2.imshow('Preview', cv2.flip(preview_image, 1))

            
            if not_calibrated :
                cv2.imshow('Face Calibration', cv2.flip(image, 1))
                
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 13 :
                if not_calibrated :
                    not_calibrated = False
                    calibration_face = calibrate_face(points_mat)
                    cv2.destroyWindow("Face Calibration")
                    avatar_ready = True
            
            if avatar_ready :
                calibration_face = calibrate_face(points_mat)
                H, homography_is_good = compute_calibration_transformations(calibration_avatar, calibration_face)
                if homography_is_good :
                    result_image = cv2.warpPerspective(result_image, H, (480, 480))
                    cv2.imshow('Result', cv2.flip(result_image, 1))

            


    cap.release()
        
    

if __name__ == '__main__':
    _main_()
    
