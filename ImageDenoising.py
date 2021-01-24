#%%  Asked in L&T interview for OpenCV opening
# NLM is mostly used in Computer Tomography CT scanners.
# NLM and Biletral filters preserves the edges of the image.
# NLM takes the average of the similaer regions weights.
# NLM replaces the value of the pixel by an average of selection 
# of other pixels values.
#
# # OpenCV reads an image as in BGR channels whereas
# # Matplotlib reads an image as RGB format.

# these three methods of denoising preserves the edges are
# Biletral Filters
# NLM Filters
# Total Variation Filters
# denoising the image some more image filteration method
# BM3D
# 
# 

import cv2
import matplotlib.pyplot as plt 

#  Image Read
img = cv2.imread('Img_2.png')
plt.imshow(img)
plt.show()
# Denoising the image using Non-Local Mean Denoising for colored image
dst = cv2.fastNlMeansDenoisingColored(img, None, 70, 10, 7, 41)
# Non-Local image sequence of gray images
dst1 = cv2.fastNlMeansDenoisingMulti(img, 2,5,None, 4, 7, 35)
plt.imshow(dst)
plt.show()
# changing the image's channels
b,g,r = cv2.split(dst)
rgb = cv2.merge([r,b,g])
plt.imshow(rgb)

# %% SMoooding the image
# 2D confolutional smoothing
ker = np.ones((5,5), np.float32)/25
dst2 = cv2.filter2D(rgb, -1, ker)
plt.imshow(dst2)

# %%
