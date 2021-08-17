# -*- coding: utf-8 -*-
"""

@author: wuming
"""
#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
import matplotlib as plt
# 通过卷积实现sobel算子
def convolution(img,imgKernel):
    h,w=img.shape
    hk,wk=imgKernel.shape
    # outImg=np.zeros((h,w),dtype='uint8')
    outImg = np.zeros((h, w))
    for i in range(int((hk+1)/2-1),int(h-(hk+1)/2)):
        for k in range(int((hk + 1) / 2 - 1), int(w - (wk + 1) / 2)):
            aa=np.sum(img[i - int((hk + 1) / 2 - 1):i + int((hk + 1) / 2 - 1) + 1,k - int((hk + 1) / 2 - 1):k + int((hk + 1) / 2 - 1) + 1] * imgKernel)
            outImg[i,k]=abs(np.sum(img[i-int((hk+1)/2-1):i+int((hk+1)/2-1)+1,k-int((hk + 1) / 2 - 1):k+int((hk + 1) / 2 - 1)+1]*imgKernel))
    return np.uint8(outImg)

if __name__=='__main__':
    image=cv2.imread("lenna.png")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    kernel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    dst=convolution(gray,kernel)#kernel是卷积模板是卷积核翻转180°后的结果
    cv2.imshow("dst",dst)
    cv2.waitKey(0)