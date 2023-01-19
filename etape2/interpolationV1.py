import cv2
import mediapipe as mp
import numpy as np
import time

def interpolate_frame(frame1, frame2, num_interp_frames, interp_type):
    """Interpolate between two frames.
    Args:
        frame1: The first frame
        frame2: The second frame
        num_interp_frames: The number of frames to interpolate between the two frames
        interp_type: The interpolation type
            cv2.INTER_NEAREST: nearest-neighbor interpolation
            cv2.INTER_LINEAR: bilinear interpolation (used by default)
            cv2.INTER_AREA: resampling using pixel area relation. It may be the best method when down-sampling an image.
            cv2.INTER_CUBIC: a bicubic interpolation over 4x4 pixel neighborhood
            cv2.INTER_LANCZOS4: Lanczos interpolation over 8x8 pixel neighborhood
            cv2.INTER_LINEAR_EXACT: bit exact bilinear interpolation
    Returns:
        interp_frames: The interpolated frames
    """
    # Create an empty list to store the interpolated frames
    interp_frames = []
    # Get the shape of the frames
    height, width = frame1.shape[:2]
    # Interpolate the values of each pixel for the specified number of frames
    for i in range(num_interp_frames):
        alpha = i / (num_interp_frames - 1)
        interp_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        interp_frame = cv2.resize(interp_frame, (width, height), interpolation=interp_type)
        interp_frames.append(interp_frame)
    # return the interpolated frame at the specified index
    return interp_frames

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(0)

# Load the avatar image from the specified path
avatar_up = cv2.imread('data/avatar/all_up.png')

avatar_up_right = cv2.imread('data/avatar/all_up_right.png')
avatar_up_left = cv2.imread('data/avatar/all_up_left.png')
avatar_right = cv2.imread('data/avatar/all_right.png')
avatar_down = cv2.imread('data/avatar/all_down.png')
avatar_down_right = cv2.imread('data/avatar/all_down_right.png')
avatar_down_left = cv2.imread('data/avatar/all_down_left.png')
avatar_left = cv2.imread('data/avatar/all_left.png')

avatar = cv2.imread('data/avatar/all_forward.png')
prev = "Looking Forward"
num_interp_frames = 100

# Create a dict that maps the text to the avatar image
avatar_dict = {'Looking Forward': avatar, 'Looking Up': avatar_up, 'Looking Up Right': avatar_up_right, 'Looking Right': avatar_right, 'Looking Down Right': avatar_down_right, 'Looking Down': avatar_down, 'Looking Down Left': avatar_down_left, 'Looking Left': avatar_left, 'Looking Up Left': avatar_up_left}

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          

            # See where the user's head tilting
            if y < -10 and x < -10:
                text = "Looking Down Left"
            elif y < -10 and x > 10:
                text = "Looking Up Left"
            elif y > 10 and x > 10:
                text = "Looking Up Right"
            elif y > 10 and x < -10:
                text = "Looking Down Right"
            elif y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Looking Forward"

            # check if text has changed
            if text != prev:
                # Interpolate the avatar image
                frames = interpolate_frame(avatar_dict[prev], avatar_dict[text], num_interp_frames, cv2.INTER_LANCZOS4)
                prev = text
                for frame in frames:
                    cv2.imshow('Avatar', frame)
                    cv2.waitKey(1)
            else:
                cv2.imshow('Avatar', avatar_dict[text])


            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        end = time.time()
        totalTime = end - start


        fps = 1
        #print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=None,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)


    cv2.imshow('Source', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()