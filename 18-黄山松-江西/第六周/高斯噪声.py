import numpy as np
import cv2
import random
from numpy import shape

def GaussianNoise(source, sigma, means, persentage):
    NoiseNum = int(persentage*source.shape[0]*source.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, source.shape[0]-1)
        randY = random.randint(0, source.shape[1]-1)
        source[randX, randY] = source[randX, randY] + random.gauss(means, sigma)
        if source[randX, randY] < 0:
            source[randX, randY] = 0
        elif source[randX, randY] > 255:
            source[randX, randY] = 255
    return source


img = cv2.imread('lenna.png', 0)
img_guass = GaussianNoise(img, 2, 4, 0.8)

img = cv2.imread('lenna.png')
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img1)
cv2.imshow('guass_noise', img_guass)
cv2.waitKey(0)

