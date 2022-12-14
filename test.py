import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the overlay image
avatar = cv2.imread('avatar_fuwa_body_parts/fuwa_glace.png')

#Check if the files opened
if  avatar is None :
    exit("Could not open the image")
if  face_cascade.empty() :
    exit("Missing: haarcascade_frontalface_default.xml")


# Create the mask for the glasses
avatarGray = cv2.cvtColor(avatar, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(avatarGray, 10, 255, cv2.THRESH_BINARY)

# Create the inverted mask for the glasses
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert glasses image to BGR
# and save the original image size (used later when re-sizing the image)
avatar = avatar[:,:,0:3]
avatarHeight, avatarWidth = avatar.shape[:2]

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened() :
    exit('The Camera is not opened')


while True:

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = face_cascade.detectMultiScale(roi_gray)

        #cv2.imshow('Video', roi_gray)
        #cv2.waitKey()

        # Center the glasses on the bottom of the nose
        x1 = x - (avatarWidth/4)
        x2 = x + w + (avatarWidth/4)
        y1 = y + h - (avatarHeight/2)
        y2 = y + h + (avatarHeight/2)

            # Check for clipping
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > w:
            x2 = w
        if y2 > h:
            y2 = h

        # Re-calculate the width and height of the glasses image
        avatarWidth = x2 - x1
        avatarHeight = y2 - y1

        # Re-size the original image and the masks to the glasses sizes
        # calcualted above
        avatar = cv2.resize(avatar, (avatarWidth,avatarHeight), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (avatarWidth,avatarHeight), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (avatarWidth,avatarHeight), interpolation = cv2.INTER_AREA)

        # take ROI for glasses from background equal to size of glasses image
        roi = roi_color[y1:y2, x1:x2]

        # roi_bg contains the original image only where the glasses is not
        # in the region that is the size of the glasses.
        
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # roi_fg contains the image of the glasses only where the glasses is
        roi_fg = cv2.bitwise_and(avatar,avatar,mask = mask)

        # join the roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)

        # place the joined image, saved to dst back over the original image
        roi_color[y1:y2, x1:x2] = dst
    #break
    #Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()