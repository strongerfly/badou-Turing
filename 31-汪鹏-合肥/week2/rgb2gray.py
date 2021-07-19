# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:48:35 2021
rgb三通道图像转化成不同的灰度图、二值化图
RBG2Gray
@author: wp
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

plt.subplot(421)
img = plt.imread("lenna.png")
plt.imshow(img)

fig = cv2.imread("lenna.png")
h, w = fig.shape[:2]
fig_gray_fd = np.zeros([h, w], fig.dtype)
fig_gray_zs, fig_gray_wy = fig_gray_fd, fig_gray_fd
fig_gray_pj, fig_gray_g = fig_gray_fd, fig_gray_fd
for i in range(h):
    for j in range(w):
        f = fig[i, j]
        fig_gray_fd[i, j] = f[2] * 0.3 + f[1] * 0.59 + f[0] * 0.11
        fig_gray_zs[i, j] = int(f[2] * 30 + f[1] * 59 + f[0] * 11) / 100
        fig_gray_wy[i, j] = (f[2] * 76 + f[1] * 151 + f[0] * 28) >> 8
        fig_gray_pj[i, j] = (f[2] * 1.0 + f[1] * 1.0 + f[0] * 1.0) / 3.0
        fig_gray_fd[i, j] = f[1]

plt.subplot(422)
plt.imshow(fig_gray_fd)

plt.subplot(423)
plt.imshow(fig_gray_zs, cmap='gray')
    
plt.subplot(424)
plt.imshow(fig_gray_wy, cmap='gray')
    
plt.subplot(425)
plt.imshow(fig_gray_pj, cmap='gray')

plt.subplot(426)
plt.imshow(fig_gray_g, cmap='gray')

""" 
# 二值化
h, w = fig_gray_fd.shape
fig_binary = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        if fig_gray_fd[i, j] >= 0.5:
            fig_binary[i, j] =  1
"""
plt.subplot(427)
fig_binary = np.where(fig_gray_fd >= 0.5, 1, 0)
plt.imshow(fig_binary, cmap='gray')
plt.show
