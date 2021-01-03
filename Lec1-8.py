#%%
import cv2 as cv 

#  Camera read for default camera (0)
cam = cv.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    #  Hue, Saturation, Value (HSV)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) 
    cv.imshow('hsv', hsv)
    lower_red = np.array([30,150,50]) 
    upper_red = np.array([255,255,180]) 
    mask = cv.inRange(hsv, lower_red, upper_red) 
    res = cv.bitwise_and(frame,frame, mask= mask) 
    cv.imshow('res', res)
    cv.imshow('Original',frame) 

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 

# %%
