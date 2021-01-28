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
# plt.imshow(red_channel, cmap='gray')

plt.hist(red_channel.flat, bins=150, range=(0,255))  # bins are average values of bins


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
cv2.imshow("Red Channel", r)
region_1 = (r >= 0) & (r < 90)
region_2 = (r > 90) & (r < 150)
region_3 = (r > 150) & (r < 200)
region_4 = (r > 200) & (r <= 255)

all_mask = np.zeros((r.shape[0], r.shape[1], 3))
all_mask[region_1] = (1,0,0) # Blue
all_mask[region_2] = (0,1,0) # Green
all_mask[region_3] = (0,0,1) # Red
all_mask[region_4] = (1,1,0) # Yellow
cv2.imshow("Segmented Image", all_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
from skimage.filters import threshold_multiotsu
thresholds1 = threshold_multiotsu(r, classes=4)
regs = np.digitize(r, bins=thresholds1)
cv2.imshow("OTSU segmentation", regs)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
