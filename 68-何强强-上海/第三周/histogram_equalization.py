# -*- coding:utf-8 -*-
"""
直方图均衡化
原文链接：https://blog.csdn.net/a13602955218/article/details/84310394
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
imag

imaes:输入的图像
channels:选择图像的通道
mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
histSize:使用多少个bin(柱子)，一般为256
ranges:像素值的范围，一般为[0,255]表示0~255
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


def func(img):
    h, w = img.shape
    input_hist, _ = np.histogram(img, 256)
    output_data = img.copy()
    pis, q_list = [], []
    for i in range(0, 256):
        pi = input_hist[i] / (h * w)
        pis.append(pi)
        q = sum(pis) * 256 - 1
        q = max(0, q)
        q = min(255, q)
        q_list.append(q)
    # todo 怎样替换？
    return output_data



if __name__ == '__main__':

    src = cv2.imread("lenna.png")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # res = cv2.calcHist(images=[gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    # # 原始图像直方图
    # plt.bar(range(0, 256), res.ravel())
    # plt.title("origin hist")
    # plt.show()
    # cv2 均衡化
    # eq1 = cv2.equalizeHist(gray)
    # plt.hist(eq1.ravel(), 256)
    # plt.title("equalization by cv2")
    # plt.show()

    des = func(gray)
    plt.hist(des.ravel(), 256)
    plt.title("equalization by cv2")
    plt.show()
