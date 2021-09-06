#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing
@File    ：noise_generator.py
@Author  ：micro
@Date    ：2021/7/28 上午10:47
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def add_gaussian_noise(gray, mean, sigma):
    """ 高斯噪声的手动实现

    :param gray: 灰度图片
    :type gray: np.ndarray
    :param mean: 均值
    :type mean: float
    :param sigma: 方差
    :type sigma: float
    :return: 增加高斯噪声的图片
    :rtype: np.ndarray
    """

    gaussian_noise = np.random.normal(mean, sigma, gray.shape)
    return gray + gaussian_noise


def add_pepper_salt(gray, snr):
    """ 椒盐噪声的手动实现

    :param gray: 灰度图片
    :type gray: np.ndarray
    :param snr: 信噪比
    :type snr: float
    :return: 增加椒盐噪声的图片
    :rtype: np.ndarray
    """
    height, width = gray.shape
    # 生成随机噪声点的X，Y坐标
    # 注：矩阵中的height对应的是Y坐标，width对应的是X坐标
    coordXSample = np.random.choice(width, int(width * snr), replace=False)
    coordYSample = np.random.choice(height, int(height * snr), replace=False)
    # 生成随机椒(0)盐(255)点
    papperSaltSample = np.random.choice([0, 255], (int(height * snr), int(width * snr)))

    target = gray.copy()
    target[coordYSample[:, np.newaxis], coordXSample] = papperSaltSample
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--path', required=True, help="image path for input")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_noise = add_gaussian_noise(gray, 2, 50)
    papper_salt_noise = add_pepper_salt(gray, 0.5)

    plt.subplot(221)
    plt.imshow(gray, cmap='gray')
    plt.subplot(222)
    plt.imshow(gray_noise, cmap='gray')
    plt.subplot(223)
    plt.imshow(papper_salt_noise, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
