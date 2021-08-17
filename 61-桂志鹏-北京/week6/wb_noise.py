# 椒盐噪声
import cv2
import random

img = cv2.imread('img/lenna.png', 0)
h,  w = img.shape
rate = 0.2
wb_img = img

noise_num = int(h * w * rate)
for i in range(noise_num):
    rand_x = random.randint(0, h - 1)
    rand_y = random.randint(0, w - 1)
    if random.random() < 0.5:
        wb_img[rand_x, rand_y] = 0
    else:
        wb_img[rand_x, rand_y] = 255
cv2.imshow('src', cv2.imread('img/lenna.png', 0))
cv2.imshow('noise', wb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
