# -*- coding:utf-8 -*-
import random

import cv2
import numpy as np


def add_salt_pepper_noise(img, percentage):
    noise_count = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noise_count):
        # 椒盐噪声图片边缘不处理，故-1
        x = random.randint(0, img.shape[0] - 1)
        y = random.randint(0, img.shape[1] - 1)
        img[x][y] = random.choice([0, 255])
    return img


def add_gaussian_noise(img, percentage, mu=0, sigma=1):
    noise_count = int(percentage * img.shape[0] * img.shape[1])
    for i in range(noise_count):
        # 高斯噪声图片边缘不处理，故-1
        x = random.randint(0, img.shape[0] - 1)
        y = random.randint(0, img.shape[1] - 1)
        n_v = img[x][y] + random.gauss(mu, sigma)
        if n_v < 0:
            n_v = 0
        if n_v > 255:
            n_v = 255
        img[x][y] = n_v
    return img


if __name__ == '__main__':
    src = cv2.imread("lenna.png", 0)
    cv2.imshow('origin', src)

    sp_img = add_salt_pepper_noise(src.copy(), 0.2)
    cv2.imshow('salt and pepper img', sp_img)

    gs_img = add_gaussian_noise(src.copy(), 0.2)
    cv2.imshow('gaussian img', gs_img)
    cv2.waitKey()
