import numpy as np
import cv2
import random

def GaussianNoise(src, means, sigma, percetage):
    noise_img = src
    noisenum = int(percetage * noise_img.shape[0] * noise_img.shape[1])
    for i in range(noisenum):
        randx = random.randint(0, noise_img.shape[0]-1)
        randy = random.randint(0, noise_img.shape[1]-1)
        noise_img[randx, randy] = noise_img[randx, randy] + random.gauss(means, sigma)
        if noise_img[randx, randy] < 0:
            noise_img[randx, randy] = 0
        elif noise_img[randx, randy] > 255:
            noise_img[randx, randy] = 255
    return noise_img

img = cv2.imread("D:/GoogleDownload/lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
img2 = GaussianNoise(img, 2, 4, 0.8)
cv2.imshow("yuantu", img1)
cv2.imshow("noise img", img2)
cv2.waitKey(0)