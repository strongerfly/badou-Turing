#!/usr/bin/python
# -*- coding: utf-8 -*-

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img=cv2.imread("lenna.png")
h,w=img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(img)
print(img_gray)
#cv2.imshow("gray image",img_gray)
#cv2.waitKey(10)
###
# 灰度化
img_gray1 = rgb2gray(img)
plt.subplot(221)
plt.imshow(img_gray1, cmap='gray')
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(222)
plt.imshow(img_gray2, cmap='gray')
plt.waitforbuttonpress(0)

