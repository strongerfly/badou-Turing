import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png',1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Gaussian blur
img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
'''
Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
image:灰度图像，单通道图像
threshold1, ：最小梯度值
threshold2,：最大梯度值
'''
mean1 = int(img_gray.mean())
v1 = cv2.Canny(img_gray, mean1 * 0.5, mean1 )
v2 = cv2.Canny(img_gray, mean1 * 0.3, mean1 * 1.5)

cv2.imshow('v1', v1)
cv2.waitKey(0)

cv2.imshow('v2', v2)
cv2.waitKey(0)
cv2.destroyAllWindows()
