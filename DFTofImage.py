#%%
import cv2
import numpy as np 

#%%  Importing the Image
# img = cv2.imread("baboon.jpg")
img = cv2.imread("baboon.jpg")
b,g,r = cv2.split(img)
cv2.imshow("Original Image", img)
cv2.imshow("Blue Channel Image", b)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%   High pass filter design of the image
# In this tutorial we are we are designing a high pass filterfor the given image
# Step 1 - Transform the image into Frequency domain.
# Step 2 - Design a High pass mask for the image. a circuler mask with radius as the cut-off 
# frequency.
# Step 3 - Multiply the mask with the Frequency transformed image.
# Step 4 - Take the Inverse DFT of the masked image.
# Step 5 - This is the High pass filtered image.
# Step 6 - Tune the radius as a Hypreparameter for the cut-off frequency. 

# Step 1
dft = cv2.dft(np.float32(b), flags=cv2.DFT_COMPLEX_OUTPUT)
# shifting the DFT
dft_shift = np.fft.fftshift(dft)
mag_spect = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Step 2
rows, cols = b.shape 
cenrow, cencol = int(rows/2), int(cols/2)
mask = np.ones((rows, cols, 2), np.uint8)
# Radius of the mask.
r = 60 
center = [cenrow, cencol]
x,y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) **2 <= r*r
mask[mask_area] = 0
cv2.imshow("Magnitude Spectrum", mag_spect)

# Step 3
fshift = dft_shift * mask

# Step 4
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Step 5
cv2.imshow("Filtered Blue channel", img_back)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
