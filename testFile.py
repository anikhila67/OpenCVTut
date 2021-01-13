#%%
import cv2 as cv 

#  Camera read for default camera (0)
cam = cv.VideoCapture(0)

# width of frame 
box_size = 234
width = int(cam.get(3))
while(1):
    ret, frame = cam.read()

    # Flipped frame 
    # fframe = cv.flip(frame, 1)
    # cv.imshow('Flipped frame', fframe)
    
    # placing a rectangle on the frame
    cv.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)

    cv.imshow('Main', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edge = cv.Canny(gray, 50, 150)
    cv.imshow('Canny', edge)

    # Median Blur
    # Mblur = cv.medianBlur(gray,5)
    # cv.imshow('Median Blur', Mblur)

    # # Gaussian blur
    # Gblur = cv.GaussianBlur(gray,(5,5),0)
    # cv.imshow('Gaussian Blur', Gblur)
    
    # # Laplacian Blur
    # # check for kernel size ksize= 3, 5, 7, etc
    # Lblur = cv2.Laplacian(frame, cv2.CV_64F, ksize=7)
    # cv.imshow('Laplacian', Lblur)

    # # Biletral Blur
    # # change different window size
    # Biletrarblur = cv.bilateralFilter(frame,9,95,95)
    # cv.imshow('Biletral Blur', Biletrarblur)

    k = cv.waitKey(5) & 0xFF
    if k == 27: 
        cam.release() 
        cv.destroyAllWindows()
        break
#%%
# camera release 
cam.release() 
# De-allocate any associated memory usage 
cv.destroyAllWindows()
# 

# %%
