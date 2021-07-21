# 1.最邻近插值实现
import cv2
import numpy as np


def near_insert(img, h, w):
    height,width,channels = img.shape
    target = np.zeros((int(height * h), int(width * w), channels), np.uint8)
    for x in range(target.shape[0] - 1):
        for y in range(target.shape[1] - 1):
            sc_x = int(x / h)
            sc_y = int(y / w)
            target[x, y] = img[sc_x, sc_y]
    return target


img_src = cv2.imread('lenna.png')
print(img_src.shape)
zoom = near_insert(img_src, 1.5, 1.5)
print(zoom.shape)
cv2.imshow('intrep', zoom)
cv2.imshow('img', img_src)
cv2.waitKey(0)