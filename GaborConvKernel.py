#%%
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

#%% 
'''
Here we first segment the image using Variance, then we use Gabor kernel.
after then we user entropy for the segmentation.
Though the Gabor is the best feature extraction kernel, but for single image it does not
work fine because you have to find these parameters for a perticular features. so if we perform 
this in ML Gabor is one of the best feature extraction kernel.
Gabor is the function of a pixel, wavelengh, angle, phase, SD, gamma.
So by varing these parameters we can detect different(infinite) features of the image.

'''
img = cv2.imread('textured1.png')
cv2.imshow("Original Image", img)
cv2.waitKey()
cv2.destroyAllWindows()


#%%  Read the image and split the chennals

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

#%% Variance filter of the Image
x,y = r.shape
k = 7
ker = np.ones((7,7))

np.filter2D()

#%% Gabor kernel

ksize = 45
theta = np.pi/4
kernel = cv2.getGaborKernel((ksize, ksize), 5.0, theta, 10.0, 0.9, 0, ktype=cv2.CV_32F)
flt_img = cv2.filter2D(g, cv2.CV_8UC3, kernel)
cv2.imshow("Gabor filtered", flt_img)

cv2.waitKey(0)
cv2.destroyAllWindows()





# %%
