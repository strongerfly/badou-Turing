#随机生成符合正态分布，means，sigma为两个参数
import numpy as np
import cv2
from numpy import shape
import random

def GaussionNoise(src,means,sigma,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        #每一个随机点
        #randX代表生成的行，randY代表生成的列
        #高斯噪声图片边缘不处理，固-1
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)

        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if NoiseImg[randX,randY]<0:
            NoiseImg[randX, randY]=0
        if NoiseImg[randX,randY]>255:
            NoiseImg[randX,randY]=255
        return NoiseImg

img =cv2.imread('lenna.png',0)
img1 = GaussionNoise(img,7,7,0.8)
img = cv2.imread('lenna.png')
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img3=GaussionNoise(img2,2,4,0.8)
cv2.imshow('src1',img)
cv2.imshow('src2',img2)
cv2.imshow('dst',img1)
cv2.imshow('dst2',img3)
cv2.waitKey(0)

