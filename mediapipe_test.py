import cv2
import mediapipe as mp

# Load the avatar image and get its dimensions
avatar_image = cv2.imread("avatar_fuwa_body_parts/fuwa_glace.png")
avatar_height, avatar_width = avatar_image.shape[:2]

# Set up the mediapipe face detection pipeline
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5)

# Open the video stream
video_capture = cv2.VideoCapture(0)

# Set up the display windows
cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)
cv2.namedWindow("Avatar Video", cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the video stream
    ret, frame = video_capture.read()

    # Detect faces in the frame
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:

        # Loop through the detected faces
        for detection in results.detections:

            # Get the face bounding box coordinates
            face_x, face_y, face_w, face_h = detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height

            # Compute the avatar scaling and rotation factors
            face_center = (face_x + face_w / 2, face_y + face_h / 2)
            face_angle = cv2.getRotationMatrix2D(face_center, 0, 1)[0][1]
            face_scale = face_w / avatar_width

            # Rotate and scale the avatar image
            avatar_rotation_matrix = cv2.getRotationMatrix2D(face_center, face_angle, face_scale)
            avatar_rotated = cv2.warpAffine(avatar_image, avatar_rotation_matrix, (avatar_width, avatar_height))

                        # Resize the avatar image to match the dimensions of the video frame
            avatar_resized = cv2.resize(avatar_rotated, (int(face_w), int(face_h)))


        # Display the original and avatar video frames
        cv2.imshow("Original Video", frame)
        cv2.imshow("Avatar Video", avatar_rotated)

    # Check for user input to stop the video
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video stream and close
video_capture.release()
cv2.destroyAllWindows()
