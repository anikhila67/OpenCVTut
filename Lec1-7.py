#%%
import cv2 as cv 

#  Camera read for default camera (0)
cam = cv.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    #  Hue, Saturation, Value (HSV)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) 
    cv.imshow('hsv', hsv)
    #  Hue, Lightness, Saturation (HLS)
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS) 
    cv.imshow('hls', hls)

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 

# %%
