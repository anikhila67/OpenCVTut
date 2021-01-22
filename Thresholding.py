#%%
import cv2 
import matplotlib.pyplot as plt 
#%%  Read the image and split the chennals

img = cv2.imread('subpixel5.jpg')
cv2.imshow('Image',img)
b,g,r = cv2.split(img)
cv2.imshow('Red Image', r)
cv2.imshow('Green Image', g)
cv2.imshow('Blue Image', b)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% draw the histogram for the red chennal alone

red_channel = img[:,:,2]
plt.imshow(red_channel, cmap='gray')

plt.hist(red_channel.flat, bins=100, range=(0,255))  # bins are average values of bins


# %%  apply manual threshold by value 60
background = (red_channel <= 60)
ct = (red_channel > 60)
plt.imshow(ct, cmap='gray')

# %%  Thresholding using cv2.threshold binary 

ret, thersh1 = cv2.threshold(red_channel, 80, 255, cv2.THRESH_BINARY)
plt.imshow(thersh1, cmap='gray')

#%%  OTSU thresholding 

ret2, thersh2 = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thersh2, cmap='gray')
#%%   DIgitizing the threshold image

import numpy as np 
region1 = np.digitize(red_channel, bins=np.array([ret2]))
plt.imshow(region1)
# %%
