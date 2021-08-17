# 实现高斯噪声
import random

import cv2

img = cv2.imread('img/lenna.png', 0)
h, w = img.shape

noise_img = img
noise_num = int(0.8 * h * w)
for i in range(noise_num):
    rand_x = random.randint(0, h - 1)
    rand_y = random.randint(0, w - 1)
    noise_img[rand_x, rand_y] = noise_img[rand_x, rand_y] + random.gauss(0, 8)  # 参数如何确定合适的范围
    if noise_img[rand_x, rand_y] < 0:
        noise_img[rand_x, rand_y] = 0
    elif noise_img[rand_x, rand_y] > 255:
        noise_img[rand_x, rand_y] = 255


img_src = cv2.imread('img/lenna.png')
img2 = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
cv2.imshow('src', img2)
cv2.imshow('gauss', noise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
