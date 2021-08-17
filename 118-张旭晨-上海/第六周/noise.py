# -*- coding: utf-8 -*-
# 导入相应的包
import cv2
import numpy as np
import random
import sys
from PIL import Image
from skimage import util


def gaussiannoise(img, mean, percent, sigma):
    dst =img.copy()
    h, w = img.shape[0:2]
    # print(h,w)
    # sys.exit(1)
    # 计算噪音数量
    noisenum = int(percent * h * w)
    # 添加噪音
    for i in range(noisenum):
        # x轴的随机数
        # 去除边界
        x = random.randint(90, 270)
        y = random.randint(210, 340)
        # 当前位置增加高斯随机数
        total = img[x, y] + random.gauss(mean, sigma)
        if total < 0:
            total = 0
        if total > 255:
            total = 255
        dst[x, y] = total
    return dst


def pepersaltnoise(img, percent):
    dst = img.copy()
    h, w = img.shape[0:2]
    noisenum = int(percent * h * w)
    for i in range(noisenum):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        rand = random.random()
        if rand < 0.5:
            dst[x, y] = 0 # peper noise
        else:
            dst[x, y] = 255 # salt noise
    return dst


'''
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
功能：为浮点型图片添加各种随机噪声
参数：
image：输入图片（将会被转换成浮点型），ndarray型
mode： 可选择，str型，表示要添加的噪声类型
	gaussian：高斯噪声
	localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
	poisson：泊松噪声
	salt：盐噪声，随机将像素值变成1
	pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
	s&p：椒盐噪声
	speckle：均匀噪声（均值mean方差variance），out=image+n*image
seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
--------
返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
'''


def noise(img, mode):
    return util.random_noise(img, mode=mode)


if __name__ == '__main__':
    img = cv2.imread('lenna.jpg', 0)
    dst = gaussiannoise(img, 200, 0.1, 400)
    dst2 = pepersaltnoise(img,  0.2)
    dst3 = noise(img, 'poisson')
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst3', dst3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()