#%% 
import cv2
import numpy as np;
im = cv2.imread("blob_detection.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Image", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
detector = cv2.SimpleBlobDetector()
keypoints = detector.detect(im)
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(
    im, 
    keypoints, 
    np.array([]), 
    (0,0,255), 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

cv2.waitKey()
cv2.destroyAllWindows()

#%%
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;

# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

#%%

