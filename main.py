import cv2
import numpy as np

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the avatar image from the specified path
avatar_image = cv2.imread('avatar_fuwa_body_parts/fuwa_glace.png')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

# Create windows
cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
cv2.namedWindow('Avatar', cv2.WINDOW_NORMAL)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the original frame
    cv2.imshow('Source', frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize the avatar image as a copy of the original avatar
    avatar_image_output = avatar_image.copy()

    # Iterate through the faces and draw a rectangle around each face
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

      # Update the position and orientation of the avatar image
      # based on the position and orientation of the face
      avatar_x = x
      avatar_y = y
      avatar_w = w
      avatar_h = h
      avatar_angle = ...

      # Resize and rotate the avatar image
      avatar_image_output = cv2.resize(avatar_image_output, (avatar_w, avatar_h), cv2.INTER_LINEAR)
      avatar_image_output = cv2.rotate(avatar_image_output, cv2.ROTATE_90_CLOCKWISE)


    # Display the avatar image in the avatar window
    cv2.imshow('Avatar', avatar_image_output)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
