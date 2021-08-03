import numpy
import cv2
import random


def PepperSalt(src, percetage):
    noise_img = src
    noisenum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(noisenum):
        randx = random.randint(0, src.shape[0]-1)
        randy = random.randint(0, src.shape[1]-1)
        if random.random() < 0.5:
            noise_img[randx, randy] = 0
        elif random.random() >= 0.5:
            noise_img[randx, randy] = 255
    return noise_img

img = cv2.imread("D:/GoogleDownload/lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
img2 = PepperSalt(img, 0.6)
cv2.imshow("yuantu", img1)
cv2.imshow("noise img", img2)
cv2.waitKey(0)