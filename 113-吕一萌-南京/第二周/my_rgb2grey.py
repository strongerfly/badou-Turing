"""
@author: lym

彩色图像的灰度化、二值化
"""

import cv2
import numpy as np


def my_imread(path) -> np.ndarray:
    return cv2.imread(path)


img = my_imread("lenna.png")
print(img)
print(img.shape)
height, width = img.shape[:2]
print(img.dtype)
img_gray = np.zeros((height, width), img.dtype)
for i in range(height):
    for j in range(width):
        b, g, r = img[i, j]
        img_gray[i, j] = int(b * 0.11 + g * 0.59 + r * 0.3)
print(img_gray)
# cv2.imshow("image show gray", img_gray)
cv2.imwrite("lenna_gray.png", img_gray)
