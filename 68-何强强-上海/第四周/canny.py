# -*- coding:utf-8 -*-
"""
opencv 提供的 canny函数
"""
import math

import cv2
import numpy as np


def convolution(data, kernel, pad_width=0):
    """
    卷积实现
    :param data:  二维数组
    :param kernel:  卷积核
    :param pad_width:  填充
    :return:
    """
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


def sobel(src):
    """使用 sobel 进行边缘检测"""
    k = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3)
    sobel_img = convolution(src, k, 1)
    cv2.imshow("sobel img", sobel_img)


def cv_canny(src):
    """使用canny算法进行边缘检测"""
    canny_img = cv2.Canny(src, 10, 50)
    cv2.imshow("canny_img", canny_img)


if __name__ == '__main__':
    img = cv2.imread("tt.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_canny(gray)
    sobel(gray)
    cv2.waitKey()
