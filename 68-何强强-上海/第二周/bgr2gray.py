# -*- coding: utf-8 -*-
"""
ps: openCV 读进来的格式为 BGR
"""
import cv2
import numpy as np


def change(img):
    height, width, channel = img.shape
    new_img = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            o = img[i][j]
            gray = o[0] * 0.11 + o[1] * 0.59 + o[2] * 0.3
            new_img[i][j] = [gray]
    return new_img


if __name__ == '__main__':
    src = cv2.imread("lenna.png")
    # 原图像
    cv2.imshow("src", src)
    des = change(src)
    cv2.imshow("change", des)
    # cv2 自带方法
    gray_img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("gray", gray_img)
    cv2.waitKey()
