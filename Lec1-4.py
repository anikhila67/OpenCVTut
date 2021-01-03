#%%
import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 

img = cv.imread('baboon.jpg')
# plt.imshow(img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.hist(gray.ravel(), 256, [0,256])
plt.show()

#%% Histogram equalization is the mapping of one distribution to another distribution.
cv.imshow('Gray',gray)
img_histogram = cv.equalizeHist(gray)
cv.imshow('Histogram',img_histogram)
cv.waitKey() 
# How you re-map the distribution for the image

# %%
cv.destroyAllWindows()
