#!/usr/bin/env python
# encoding=gbk

'''
Canny边缘检测：优化的程序
'''
import cv2
import numpy as np

def gauss_noise(sigma, mu=0):
    h, w = gray.shape
    noise = sigma * np.random.randn(h, w) + mu
    new_gray = gray + noise
    new_gray[new_gray > 255] = 255
    new_gray[new_gray < 0] = 0
    cv2.imshow('gauss noise demo',new_gray.astype(np.uint8))

sigma = 5
mu = 0
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #转换彩色图像为灰度图
cv2.namedWindow('gauss noise demo')

#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用(的回调函数名
'''
# cv2.createTrackbar('sigma','gauss noise demo',sigma, 100, gauss_noise)
cv2.createTrackbar('mu','gauss noise demo',mu, 100, gauss_noise)
gauss_noise(0)  # initialization
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()
