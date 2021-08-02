import random
import cv2
import numpy as np


def channel_gauss_noice(src_image, noice_num, mu, sigma):
    # 对每个通道做高斯噪声
    channel_gauss_image = src_image
    i = 0
    while i < noice_num:
        rand_x = random.randint(0, src_image.shape[0] - 1)
        rand_y = random.randint(0, src_image.shape[1] - 1)
        # 利用公式计算高斯噪声
        channel_gauss_image[rand_x, rand_y] = src_image[rand_x, rand_y] + random.gauss(mu, sigma)
        i += 1
    return channel_gauss_image


def gauss_noice(src_image, percent, mu, sigma):
    noice_num = src_image.shape[0] * src_image.shape[1] * percent
    if len(src_image.shape) == 2:
        # 灰度图像
        return channel_gauss_noice(src_image, noice_num, mu, sigma)
    elif len(src_image.shape) > 2:
        # RGB图像
        rgb_gauss_image = []
        for i in range(src_image.shape[2]):
            # 对每个channel循环处理
            rgb_gauss_image.append(channel_gauss_noice(src_image[:, :, i], noice_num, mu, sigma))
        return np.dstack(rgb_gauss_image)


# 加载RGB原图
image = cv2.imread('lenna.png')

# 灰度图
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', image_gray)
cv2.waitKey(0)
# 对灰度图做高斯噪声
gray_gauss_image = gauss_noice(image_gray, 0.8, 1, 0.5)
cv2.imshow('gray gauss noice', gray_gauss_image)
cv2.waitKey(0)

# 对RGB图做高斯噪声
rgb_gauss_image = gauss_noice(image, 0.8, 1, 0.5)
cv2.imshow('rgb gauss noice', rgb_gauss_image)
cv2.waitKey(0)
