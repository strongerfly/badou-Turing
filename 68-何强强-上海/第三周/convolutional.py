# -*- coding:utf-8 -*-
"""
卷积原理
"""
import math
import cv2
import numpy as np


def convolution(data, kernel, pad_width=0):
    data = np.pad(data, pad_width)
    if len(data.shape) != 2 or len(kernel.shape) != 2:
        raise ValueError("需要二维数组")
    dx, dy = data.shape
    kx, ky = kernel.shape
    radius = math.floor(kx / 2)
    result = np.zeros(data.shape, dtype=np.uint8)
    for x in range(dx - kx + 1):
        for y in range(dy - ky + 1):
            result[x + radius, y + radius] = np.nansum(data[x: x + kx, y:y + ky] * kernel)
    return result


def con_test():
    o_d = np.arange(16).reshape(4, 4)
    k = np.array([1, 0, -1, 1, 0, -1, 1, 0, -1]).reshape(3, 3)
    n_d = convolution(o_d, k)
    print(o_d)
    print("*" * 100)
    print(n_d)
    n_d2 = convolution(o_d, k, 1)
    print("*" * 100)
    print(n_d2)


if __name__ == '__main__':
    # con_test()
    src = cv2.imread("tt.jpg")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray img", gray)
    k = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3).T
    sobel_img = convolution(gray, k, 1)
    cv2.imshow("sobel img", sobel_img)
    cv2.waitKey()