#%%  Asked in L&T interview for OpenCV opening
# Here in the image a gray image is merged with two noisy channels.
# So I split the image and show the single channel.

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
cv2.imshow("Original Image",img)
B, G, R = cv2.split(img)
cv2.imshow("Blue Channel of Image", B)

# Denoising the image using Non-Local Mean Denoising for colored image
dst = cv2.fastNlMeansDenoisingColored(img, None, 70, 10, 7, 41)
# Non-Local image sequence of gray images
dst1 = cv2.fastNlMeansDenoisingMulti(img, 2,5,None, 4, 7, 35)
cv2.imshow("NLM of G Channel",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% SMoooding the image
# 2D confolutional smoothing
ker = np.ones((5,5), np.float32)/25
dst2 = cv2.filter2D(rgb, -1, ker)
plt.imshow(dst2)

# %%
