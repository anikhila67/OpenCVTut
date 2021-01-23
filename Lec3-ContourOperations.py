#%%
import imutils
import numpy as np
import cv2
# skimage can also be used for the image preocessing library inplace of the 
# OpenCV. 
# if you use skimage to load an image, it normalize the image itself within (0,1)
# and uses float64 as the datatype.
#  
# Finding the number of contours in an image
# crop the contour area
frame = cv2.imread('Shapes.jpg')
cv2.imshow('Image', frame)
# cv2.waitKey()
# cv2.destroyAllWindows()
#  Gray scale of the image
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#  Edges of the image using Canny
edged = cv2.Canny(gray, 50, 150)
cv2.imshow('Edged', edged)
# Gaussian Blurr of the gray image
Gblur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow('Gaussian Blur', Gblur)
# apply the threshold for the binary image
thresh = cv2.threshold(Gblur, 226, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('Threshold', thresh)
# Find the contours in the image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) #calculates the center of the contour
# print the contours
# print("Contours : {}".format(cnts))
print("Number of contours in the image : ",len(cnts))
for c in cnts:
    print("Area of the contour : {}".format(cv2.contourArea(c)))
    # Draw the contours over the blobs or segmented area
    cv2.drawContours(frame, [c], 0, (255,255,255), 3)
    # To find the center of the contours
    M = cv2.moments(c)
    x = int(M["m10"]/M["m00"])
    y = int(M["m01"]/M["m00"])
    print("Center of the contour coordinates : ({},{})".format(x,y))
    cv2.circle(frame, (x,y), 7, (0,255,255), 3)

    # polynomial predict
    eps = 0.1*cv2.arcLength(c, True)
    app = cv2.approxPolyDP(c, eps, True)
    print("Approx curve is : ",app)
    p = app.shape
    if p[0] == 3:
        print("Its a Triangle")
    elif p[0] == 4:
        print("Its a Rectangle")
    elif p[0] == 5:
        print("Its a Pentagon")
    elif p[0] == 6:
        print("Its a Hexagon")
    elif p[0] == 7:
        print("Its a Tetragonal")
    elif p[0] == 8:
        print("Its a Octagonal")

cv2.imshow('contour drawn', frame)
cv2.waitKey()
cv2.destroyAllWindows()
# %%
