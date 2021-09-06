# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 21:16:29 2021
bugs: 出来的灰度图和处理后的图像一样？？ 作为实参（？）传递进去了，也改变了
在函数中编写过程中与最终输出关联，而不是在最后赋值？

@author: wp
"""

import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

def qualizeHist(img):
    h,w = img.shape
    im = img
    figs = np.zeros([h, w], img.dtype)
    ndarry = im.ravel()
    hist = np.zeros((256,1))
    histk = np.zeros((256,1))
    q = np.zeros((256,1), dtype= np.uint8)
    for i in range(len(ndarry)):   # 对应像素值的像素元数量统计
        p = ndarry[i]
        hist[p] += 1
         
    for k in range(256):   # 累加
        if k == 0:
            histk[0] = hist[0]
        else:
            histk[k] = histk[k-1] + hist[k]
            
        if histk[k] <= 0:   # 均衡化
            q[k] = 0               
        else:
            q[k] = histk[k] / (h* w)* 256 - 1
    
    # for i in range(len(ndarry)):
    #     p = ndarry[i]
    #     ndarry[i] = q[p]
        
    for i in range(h):
        for j in range(w):
            figs[i,j] = q[img[i,j]]
        
        
    # figs = np.reshape(ndarry,  [h, w])
    
    return figs
    
    
if __name__ == '__main__':
    start = time.time()
    imgs = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("original gray",gray)
    # 
    dst = qualizeHist(gray)
    
    # gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("After eq gray",gray)
    plt.figure()
    plt.hist(dst.ravel(),256)
    plt.show()

    print("Time:",time.time()-start)
    cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
    cv2.waitKey(10)
