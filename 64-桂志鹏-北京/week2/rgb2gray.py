import cv2
import numpy as np
from skimage.color import rgb2gray


img = cv2.imread("lenna.png")
height, width = img.shape[:2]

img_gray = np.zeros((height, width), img.dtype)

for h in range(height):
    for w in range(width):
        m = img[h, w]
        img_gray[h, w] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)


cv2.imshow("img", img)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)