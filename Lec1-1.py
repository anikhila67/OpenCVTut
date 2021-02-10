#%%
import cv2 as cv
import matplotlib.pyplot as plt 
#%%  Image read & image show
img = cv.imread('baboon.jpg')
cv.imshow('Baboon.jpg',img)
cv.waitKey()

# %%   Image write
cv.imwrite('baboon1.jpg', img)
cv.destroyAllWindows()


# %% convert the image into gray image
# 1 method
img1 = cv.imread("baboon.jpg")
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# 2 method
img2 = cv.imread("baboon.jpg", 0)

cv.imshow("Image1", gray)
cv.imshow("Image2", img2)

cv.waitKey(0)
cv.destroyAllWindows()
# %%
