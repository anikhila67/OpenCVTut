#%%  
import cv2 as cv 

#%%  Camera read for default camera (0)
cam = cv.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    cv.imshow('camera', frame)

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()

