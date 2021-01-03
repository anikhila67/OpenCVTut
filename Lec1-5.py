#%%
import cv2 as cv 

#  Camera read for default camera (0)
cam = cv.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)
    # Gray Canny Edges
    Graycanny = cv.Canny(gray, 100, 200, L2gradient=150)
    cv.imshow('Gray Canny',Graycanny)

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 
