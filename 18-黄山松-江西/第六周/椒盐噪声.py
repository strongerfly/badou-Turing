import cv2
import numpy as np
import random
from numpy import shape


def PepperSalt(source, percentage):
    NoiseImg = source
    NoiseNum = int(percentage*NoiseImg.shape[0]*NoiseImg.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, NoiseImg.shape[0]-1)
        randY = random.randint(0, NoiseImg.shape[1]-1)
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        elif random.random() > 0.5:
            NoiseImg[randX, randY] = 255
    return NoiseImg

img = cv2.imread('lenna.png', 0)
img_ps = PepperSalt(img, 0.2)

img = cv2.imread('lenna.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img1)
cv2.imshow('PepperSalt', img_ps)
cv2.waitKey(0)




