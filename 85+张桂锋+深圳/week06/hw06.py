'''
第六周作业：
1）实现高斯/椒盐噪声
'''
import numpy as np
import cv2
import random

#高斯噪声
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg

#椒盐噪声
def SaltPepperNoise(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.random() <= 0.5:
            NoiseImg[randX,randY]= 0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = GaussianNoise(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 4, 0.8)
    img3 = SaltPepperNoise(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),0.2)
    cv2.imshow('socrce_img', img2)
    cv2.imshow('lenna_GaussianNoise', img1)
    cv2.imshow('SaltPepperNoise', img3)
    cv2.waitKey(0)
