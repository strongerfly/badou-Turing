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

img = cv2.imread("lenna.png")

img_shape_2 = img.shape[:2]
print(img_shape_2)
zeros_img = np.zeros(img_shape_2, img.dtype)
for i in range(img_shape_2[0]):
    for j in range(img_shape_2[1]):
        bgr = img[i,j]
        zeros_img[i,j] = bgr[2]*0.3 + bgr[1]*0.59 + bgr[0]*0.11


print(zeros_img)
# cv2.imshow("gray_image", zeros_img)



lena=plt.imread("lenna.png")
def myrgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    # return np.dot(rgb[..., :3], [0.3, 0.59, 0.11])


# print(lena[..., :3])
gray = myrgb2gray(lena)
print("my gray img:%s "%gray)
# 也可以用
# plt.imshow(gray, cmap=plt.get_cmap('gray'))
# plt.imshow(gray, cmap='Greys_r')
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()


im = Image.open('lenna.png') # 这里读入的数据是 float32 型的，范围是0-1
print("PIL image input:%s"%im)
L = im.convert('L')
# plt.show()

