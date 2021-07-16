#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("../lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def Hist (image,nchannel):
    cv2.imshow("src", image)
    if nchannel == 3:
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))
        cv2.imshow("rgb", result)

    else:
        #灰度图均衡化
        cv2.imshow('gray', cv2.equalizeHist(gray))
    cv2.waitKey(0)

Hist(img,3)
Hist(gray,1)





