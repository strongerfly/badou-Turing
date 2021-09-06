# 最邻近差值法
# 1、老师的代码

import cv2
import numpy as np

'''
def function(img):
    height, width, channels = img.shape  # 宽、高、通道数
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
'''


# 2、我写的代码
def function(img):
    """
    实现最邻近差值
    :param img: 原始图像(彩色)
    :return: 返回最邻近插值后的图像
    """
    # 高（长）、宽及通道数
    height, width, channels = img.shape
    new_image = np.zeros(shape=(750, 700,channels), dtype=np.uint8)
    sh = height / 750
    sw = width / 700
    for i in range(750):
        for j in range(700):
            x = int(sh * i)
            y = int(sw * j)
            new_image[i, j] = img[x, y]
    return new_image


img = cv2.imread('lenna.png')
new_img = function(img)
print(img.shape)#(512, 512, 3)
print(new_img.shape)#(750, 700, 3)
cv2.imshow('New_image', new_img)
cv2.imshow('init_image', img)
cv2.waitKey()


