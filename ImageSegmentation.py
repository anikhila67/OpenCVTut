#%%
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from skimage.color import label2rgb, rgb2gray
import pandas as pd 
from skimage import io, measure

#%%
# s_img = cv2.imread('subpixel5.jpg', 0)
s_img = cv2.imread('tissus.jpg', 0)
s_img = cv2.resize(s_img, (780,540))
cv2.imshow('Steel Surface', s_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%reading image using skimage

img = io.imread("SteelIron1.jpg")
plt.imshow(img)

# %%

plt.hist(s_img.flat, bins=100, range=(0,255))

# %% background and foreground thresholding subtractions

bg = (s_img < 90)
fg = (s_img > 90)
plt.imshow(bg, cmap='gray')

#%%  Median blurr the image to remove the small blobs.

ret2, thersh2 = cv2.threshold(s_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Thresholding", thersh2)
# g_gray = cv2.GaussianBlur(thersh2, (5,5), 0)
m_gray = cv2.medianBlur(thersh2, 5)
cv2.imshow("Median Gray", m_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% border cleaning


#%% Assigning the labels to the similer blobs
label_img = measure.label(edge_blob_remove, connectivity=s_img.ndim)
plt.imshow(label_img, cmap='gray')
#%%
img_tool = label2rgb(label_img, image=m_gray)
plt.imshow(img_tool)

props = measure.regionprops_table(label_img, m_gray, properties=['label',
                                                                'area',
                                                                'equivalent_diameter',
                                                                'mean_intensity',
                                                                'solidity'])

df = pd.DataFrame(props)
print(df.head())

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Simple Blob detector
can_img = cv2.imread("tissus.jpg")
b,g,r = cv2.split(can_img)

params = cv2.SimpleBlobDetector_Params()

# Define thresholds
#Can define thresholdStep. See documentation. 
params.minThreshold = 0
params.maxThreshold = 255

# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 50000

# Filter by Color (black=0)
params.filterByColor = False  #Set true for cast_iron as we'll be detecting black regions
params.blobColor = 0

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
params.maxCircularity = 1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
params.maxConvexity = 1

# Filter by InertiaRatio
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 1

# Distance Between Blobs
params.minDistBetweenBlobs = 0

# Setup the detector with parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(m_gray)

print("Number of blobs detected are : ", len(keypoints))


# Draw blobs
img_with_blobs = cv2.drawKeypoints(m_gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_with_blobs)
cv2.imshow("Keypoints", img_with_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%  erode the m_gray image to remove the small and noisy blobs
# Morphological noise removel
p_area = 0.454
ker = np.ones((3,3), np.uint8)
op_img = cv2.morphologyEx(thersh2, cv2.MORPH_OPEN, ker, iterations=2)

#  the border touching blobs can skew the result to remove them.

from skimage.segmentation import clear_border
edge_blob_remove = clear_border(op_img)
cv2.imshow("Edge Blobs Removed", edge_blob_remove)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%  dilate the image
sure_bg = cv2.dilate(op_img, ker, iterations=4)
cv2.imshow("Image sure background", sure_bg)

#%% Distance transformation
d_trans = cv2.distanceTransform(op_img, cv2.DIST_L2, 5)
cv2.imshow("Distance Transform", d_trans)
print("Distance Tansform max value : ", d_trans.max())
#%% 
#Let us threshold the dist transform by starting at 1/2 its max value.
#gives about 21.9
ret2, sure_fg = cv2.threshold(d_trans,0.5*d_trans.max(),255,0)
plt.imshow(sure_fg, cmap='gray')

#%%
#Later you realize that 0.25* max value will not separate the cells well.
#High value like 0.7 will not recognize some cells. 0.5 seems to be a good compromize

# Unknown ambiguous region is nothing but bkground - foreground
sure_fg = np.uint8(sure_fg)  #Convert to uint8 from float
unknown = cv2.subtract(sure_bg,sure_fg)
plt.imshow(unknown, cmap='gray')

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
#Connected components labeling scans an image and groups its pixels into components 
#based on pixel connectivity, i.e. all pixels in a connected component share 
#similar pixel intensity values and are in some way connected with each other. 
#Once all groups have been determined, each pixel is labeled with a graylevel 
# or a color (color labeling) according to the component it was assigned to.
ret3, markers = cv2.connectedComponents(sure_fg)
plt.imshow(markers)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 10 to all labels so that sure background is not 0, but 10
markers = markers+10

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
markers = cv2.watershed(img,markers)

#Let us color boundaries in yellow. 
#Remember that watershed assigns boundaries a value of -1
img[markers == -1] = [0,255,255]  

#label2rgb - Return an RGB image where color-coded labels are painted over the image.
img2 = color.label2rgb(markers, bg_label=0)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
