#%%
# Edge Detection steps in the Image
# 1. Noise Reduction (Default Gaussian)
# 2. Gradient calculation (4 direction gradient(Horizontal, vertical, two diagonal(eg. Sobel)))
# 3. Non-maximum suppression (find pixel with max value)
# 4. Double threshold (to obtain strong, weak, irrelevant pixel)
# 5. Edge tracking by hystersis 

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
    # Gaussian Blur
    Gblur = cv.GaussianBlur(gray,(5,5),0)
    GCanny = cv.Canny(Gblur,100, 200)
    cv.imshow('Gaussian Gray Canny',GCanny)
    
    # Median Blur
    Mblur = cv.medianBlur(gray,5)
    MCanny = cv.Canny(Mblur,100, 200)
    cv.imshow('Median gray Canny',MCanny)
    # RGB Canny
    edges = cv.Canny(frame,100,200) 
    cv.imshow('RGB original Canny',edges) 
    
    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 

# %%
