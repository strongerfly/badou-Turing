# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]                               #获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
img_plt = np.zeros([h,w,3],img.dtype)                   #创建一张和当前图片大小一样的3通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
        img_plt[i,j] = [m[2],m[1],m[0]]         # 将BGR坐标转化为RGB
# print(img_gray)
print("lenna show img_plt:")
# cv2.imshow("", img_gray)
plt.subplot(2, 2, 1)
plt.imshow(img_plt)

# plt.subplot(2, 2, 2) #第一个参数代表子图的行数；第二个参数代表该行图像的列数； 第三个参数代表每行的第几个图像。
# img = plt.imread("lenna.png")
# # img = cv2.imread("lenna.png", False)
# plt.imshow(img)
# print("lenna show img:")
# print(img)

# 灰度化
# img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap='gray')   #表示按照灰度显示
print("---image gray----")
print(img_gray)

# 二值化
# rows, cols = img_gray.shape
img_2 = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        if (img_gray[i, j] <= 128):
            img_2[i, j] = 0
        else:
            img_2[i, j] = 1
 
# img_binary = np.where(img_gray >= 0.5, 1, 0)
print("img_2")
print(img_2)
# print(img_binary.shape)
#
plt.subplot(2, 2, 3)
plt.imshow(img_2, cmap='gray')

# 8值化
img_10 = np.zeros([h, w], img.dtype)

for i in range(h):
    for j in range(w):
        if img_gray[i, j] <= 32:
            img_10[i, j] = 0
        elif img_gray[i, j] <= 64:
            img_10[i, j] = 32
        elif img_gray[i, j] <= 96:
            img_10[i, j] = 64
        elif img_gray[i, j] <= 128:
            img_10[i, j] = 96
        elif img_gray[i, j] <= 160:
            img_10[i, j] = 128
        elif img_gray[i, j] <= 192:
            img_10[i, j] = 160
        elif img_gray[i, j] <= 224:
            img_10[i, j] = 192
        else:
            img_10[i, j] = 255

        # img_10[i, j] = img_gray[i, j] / 256.0;

# img_10[0, 0] = 0.1
# img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_10------")
print(img_10)
# print(img_binary.shape)
#
plt.subplot(2, 2, 4)
plt.imshow(img_10, cmap='gray')

plt.show()
