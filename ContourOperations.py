#%%
import imutils
import numpy as np
import cv2

# Finding the number of contours in an image
# crop the contour area
frame = cv2.imread('Shapes.jpg')
cv2.imshow('Image', frame)
cv2.waitKey()
cv2.destroyAllWindows()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) #calculates the center of the contour
print("Number of contours in the image : ",len(cnts))
cnt = cnts[4]
cv2.drawContours(frame, [cnt], 0, (0,255,255), 3)
cv2.imshow('contour draw', frame)
cv2.waitKey()
cv2.destroyAllWindows()
# %%
