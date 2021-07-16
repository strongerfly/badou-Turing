# -- coding: utf-8 --
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# 灰度化
img = cv2.imread("lenna.png")
#获取图片的high和wide
h,w = img.shape[:2]
#创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        # 取出当前high和wide中的BGR坐标
        m = img[i,j]
        # 将BGR坐标转化为gray坐标并赋值给新图像
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print("-----gray------")
print (img_gray)
print (img_gray.shape)
cv2.imshow("gray", img_gray)
# 二值化
img_binary = np.where(img_gray>= 127, 255, 0)
print("-----binary------")
print(img_binary)
print(img_binary.shape)
cv2.imshow("binary", img_binary)
cv2.waitKey(0)