# -*- coding: utf-8 -*-
"""

@author: wuming
"""
#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

def hisogramEqualization(image):
    if 2!=image.ndim:
        print("请输入二维矩阵")
        return 0
    hist=cv2.calcHist([image],[0],None,[256],[0,256])
    h,w=image.shape
    outImage=np.zeros((h,w),dtype="uint8")#图像的矩阵定义时必须定义其数值类型
    pi_list=[]
    for i in range(256):
        pi=hist[i]/(h*w)
        pi_list.append(pi)
        outTempHist=np.round((sum(pi_list))*256-1)
        outTempHist=max(0, outTempHist)
        outTempHist=min(255,outTempHist)
        outImage[image==i]=outTempHist
        # np.where(image==i,outTempHist,image)

    return outImage
if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = hisogramEqualization(gray)
    cv2.imshow("dst", np.hstack([gray, dst]))
    hist=cv2.calcHist([dst],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.show()





# 获取灰度图像
# img = cv2.imread("lenna.png", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)




# 灰度图像直方图均衡化
# dst = cv2.equalizeHist(gray)
#
# # 直方图
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#
# plt.figure()
#
# plt.plot(hist)
# # plt.hist(gray.ravel(), 256)
# # plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)



# 彩色图像直方图均衡化
# img = cv2.imread("lenna.png", 1)
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)
