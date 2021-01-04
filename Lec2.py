#%%
import cv2 as cv 

#  Camera read for default camera (0)
cam = cv.VideoCapture(0)
while(1):
    ret, frame = cam.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Median Blur
    Mblur = cv.medianBlur(gray,5)
    cv.imshow('Median Blur', Mblur)

    # Gaussian blur
    Gblur = cv.GaussianBlur(gray,(5,5),0)
    cv.imshow('Gaussian Blur', Gblur)
    
    # Laplacian Blur
    # check for kernel size ksize= 3, 5, 7, etc
    Lblur = cv2.Laplacian(frame, cv2.CV_64F, ksize=7)
    cv.imshow('Laplacian', Lblur)

    # Biletral Blur
    # change different window size
    Biletrarblur = cv.bilateralFilter(frame,9,95,95)
    cv.imshow('Biletral Blur', Biletrarblur)

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        break

# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 

# %%
