import cv2
import numpy as np

img = cv2.imread("lenna.png")
height,width =img.shape[:2]
img_gray = np.zeros([height,width],img.dtype)
for i in range(height):
    for j in range(width):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print(img_gray)

cv2.imshow("gray",img_gray)
cv2.waitKey()