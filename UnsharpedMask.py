#%%  unsharpened mask = original + a*(original-blurred)
# subtracts an unsharp, or smoothed, version of an image from the original image which enhances
# the edges (high frequency components of the image)
import cv2 
import numpy as np 
#%%

img = cv2.imread('Shapes.jpg')
cv2.imshow("Shapes", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%  gaussian blur
cv2.imshow("Shapes Gray", gray)
Gblur = cv2.GaussianBlur(gray, (5,5), 0)
img2 = (gray - Gblur)*2
img3 = gray + img2

cv2.imshow("Unsharpened Mask", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
