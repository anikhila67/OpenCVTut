#%%
import cv2
import numpy as np 

#%%
car = cv2.imread("tesla.jpg")
car = cv2.resize(car, (180,120))

mars = cv2.imread("marspic.jpg")
car_mask = np.zeros(car.shape, car.dtype)
r_points = np.array([[11,109],[3,62],[17,16],[98,2],[175,14],[159,63],[70,118]], np.int32)

cv2.fillPoly(car_mask, [r_points], (255,255,255))

center = (500, 300)
output = cv2.seamlessClone(car, mars, car_mask, center, cv2.NORMAL_CLONE)


cv2.imshow("Tesla on Mars", output)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%