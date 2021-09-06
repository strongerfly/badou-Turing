# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 23:30:34 2021

@author: wp
"""
import cv2
import numpy as np

img = cv2.imread("lenna.png")

h, w, c = img.shape
h1, w1 = 800,800
sh = h / h1
sw = w / w1

img_resize = np.zeros([h1, w1, c], np.uint8)

for i1 in range(h1):
    for j1 in range(w1):
        i = int(i1 * sh)
        j = int(j1 * sw)
        img_resize[i1, j1] = img[i, j]

cv2.imshow("img",img)
cv2.imshow("img_resize_n",img_resize)
cv2.waitKey(0)
