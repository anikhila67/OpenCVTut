#%%
import cv2 as cv
import matplotlib.pyplot as plt 
#%%  Image read & image show
img = cv.imread('baboon.jpg')
plt.imshow(img)

# %%   Image write
cv.imwrite('baboon1.jpg', img)


