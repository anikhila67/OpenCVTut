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


# %%
