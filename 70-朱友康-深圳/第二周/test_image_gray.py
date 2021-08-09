# -*- coding: utf-8 -*-

"""
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

def rgb2gray():
    # 灰度化
    img = cv2.imread("../images/lenna.png")
    h,w = img.shape[:2]
    img_gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img_gray_1 = np.zeros([h,w],img.dtype)
    t1 = time.time()
    #逐像素处理
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            img_gray_1[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    t2=time.time()

    #numpy运算
    factors = np.array([0.11,0.59,0.3],dtype=np.float32)
    img_gray_2= np.clip(np.sum(img * factors,axis=2).astype(np.uint8),0,255)
    t3 = time.time()

    plt.subplot(221)
    plt.imshow(img[:,:,::-1])
    plt.subplot(222)
    plt.imshow(img_gray,cmap='gray')
    plt.subplot(223)
    plt.imshow(img_gray_1,cmap='gray')
    plt.subplot(224)
    plt.imshow(img_gray_2, cmap='gray')
    info = "逐像素循环计算耗时%.3fs，numpy api运算耗时%.3fs"%(t2-t1,t3-t2)
    print(info)
    plt.show()

if __name__=="__main__":
    rgb2gray()
