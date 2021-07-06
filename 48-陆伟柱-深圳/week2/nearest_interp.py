# -*- coding: utf-8 -*-

import cv2
import numpy as np

def function(img):
    height, width, channels = img.shape
    new_size = 800
    emptyImage = np.zeros((new_size, new_size, channels), np.uint8)
    sh = new_size / height
    sw = new_size / width
    for i in range(0, new_size, 1):#i, j交换位置后图片会发生顺时针90°旋转，步长改变图片清晰度下降
        for j in range(0, new_size, 1):
            x = int(round(i / sh, 0))
            y = int(round(j / sw, 0))
            emptyImage[i, j] = img[x, y]
    '''for i in range(new_size-1, 0, -1):#图像分割
        for j in range(new_size-1, 0, -1):
            x = int(i / sh) - 256
            y = int(j / sw) - 256
            emptyImage[i, j] = img[x, y]'''
    '''for i in range(new_size - 1, -1, -1):#图像倒置、ij交换逆时针旋转
        for j in range(new_size - 1, -1, -1):
            x = - int(round(i / sh, 0))
            y = - int(round(j / sw, 0))
            emptyImage[i, j] = img[x, y]'''
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img)
cv2.imwrite('nearest_interp_lenna.jpg', zoom)
#print(zoom)
#print(zoom.shape)
#cv2.imshow("nearest interp", zoom)
#cv2.imshow("image", img)
#cv2.waitKey(0)

