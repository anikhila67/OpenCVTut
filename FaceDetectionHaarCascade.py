#%%
import cv2
import sys

# The xml for the face is loaded
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# faceCascade = cv2.CascadeClassifier('LBPCascade/lbpcascade_frontalface.xml')
# eyes_cascade = cv2.CascadeClassifier('haarCascade/haarcascade_eye_tree_eyeglasses.xml')
# smile_cascade = cv2.CascadeClassifier('haarCascade/haarcascade_smile.xml')

#%%
video_capture = cv2.VideoCapture(0)

img_counter = 0
img_countter = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # For rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # For circle
        # center = (x + w//2, y + h//2)
        # cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        
        # To store the frame rectangle working
        roi_face = gray[y:y+h, x:x+w]
        if k%256 == 32:
            cv2.imwrite("yourPics/your_Pics{}.png".format(img_countter), roi_face)
            img_countter += 1

        # To detect the eyes
        # eyes = eyes_cascade.detectMultiScale(roi_face)
        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        # smile = smile_cascade.detectMultiScale(roi_face)
        # for (x3, y3, w3, h3) in smile:
        #     smile_center = (x + x3, y + y3)
        #     frame = cv2.rectangle(frame, (x + x3, y + y3), (x + x3 + w3, y + y3 + h3), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)

    if k%256 == 27: #ESC Pressed
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "yourPics/facedetect_webcam_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# %%
video_capture.release()
cv2.destroyAllWindows()


# %%
