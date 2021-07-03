# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
import sys
import numpy
from skimage.color import rgb2gray as skrgb2gray


# 灰度化，二值图
def rgb2gray(img):
    h, w = img.shape[:2]
    img_gray = numpy.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            # 浮点算法          B(0.11)           G(0.59)       R(0.3)
            # img_gray[i, j] = b * 0.11 + g * 0.59 + r * 0.3
            # 移位法          B(28)           G(151)       R(76)  >> 8
            # img_gray[i, j] = (b*28+g*151+r*76) >> 8
            # print(img[i, j], img_gray[i, j])
            # sys.exit()
            # 平均值
            # img_gray[i, j] = b/3 + g/3 + r/3
            # 仅取绿值
            img_gray[i, j] = g
    # plt.imshow(img_gray, cmap="gray")
    # plt.show()
    # print(img_gray)
    # cv2.imshow('new', img_gray)
    # cv2.waitKey(0)
    return img_gray


# 临近插值
def nearest_interpolation(img, newShape):
    h, w = img.shape[:2]
    nh, nw = newShape
    img_new = numpy.zeros((nh, nw, 3), img.dtype)
    for i in range(nh):
        for j in range(nw):
            img_new[i, j] = img[int(i*h/nh), int(j*w/nw)]
            # print(img[int(i*h/nh), int(j*w/nw)], int(i*h/nh), int(j*w/nw))
            # sys.exit()
    return img_new


# 双线性插值 实现
def bilinear_interpolation(img, newShape):
    h, w, c = img.shape
    nh, nw = newShape
    img_new = numpy.zeros((nh, nw, c), numpy.uint8)
    scale_x, scale_y = float(h)/nh, float(w)/nw

    for i in range(c):
        for x in range(nh):
            for y in range(nw):
                # px, py = x*scale_x, y*scale_y
                px, py = (x+0.5)*scale_x - 0.5, (y+0.5)*scale_y - 0.5
                x_o = numpy.floor(px)
                x_i = min(x_o + 1, h-1)
                y_o = numpy.floor(py)
                y_i = min(y_o + 1, w - 1)
                u = px - x_o
                v = py - y_o
                # f(i+ u, j + v) =
                # (1 - u) * (1 - v) * f(i, j) +
                # (1 - u) * v * f(i, j + 1) +
                # u * (1 - v) * f(i + 1,j) +
                # u * v * f(i + 1, j + 1)
                img_new[x, y, i] = (1-u)*(1-v)*img[int(x_o), int(y_o), i] + (1 - u) * v * img[int(x_o), int(y_i), i] + u * (1 - v) * img[int(x_i), int(y_o), i] + u * v * img[int(x_i), int(y_i), i]
    return img_new


if __name__ == '__main__':
    # img = plt.imread('./lenna.jpg')
    # cv2的文件目录下不能出现中文路径，否则会产生无法读取的NoneType
    # 相对路径读不到，可看working directory 是否为当前目录
    img = cv2.imread('lenna.jpg')
    # 自己实现二值图
    # img_gray = rgb2gray(img)
    # ski-inmage 实现二值图
    # img_gray = skrgb2gray(img)
    # 临近插值
    # img_gray = nearest_interpolation(img, [100,100])
    # 双线性插值 实现
    img_gray = bilinear_interpolation(img, [800, 600])
    # plt.imshow(img_gray, cmap="gray")
    # plt.show()
    # print(img_gray)
    print('star...')
    cv2.imshow('new', img_gray)
    cv2.waitKey(0)
