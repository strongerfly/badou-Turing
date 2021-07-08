# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('E:/lenna.png')
#灰度图像直方图及直方图均衡化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst_gray_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
dst_gray_histEqualization = cv2.equalizeHist(gray)

#绘制灰度图像直方图
plt.figure(1)
plt.title('gray_hist')
plt.xlabel('bins')
plt.ylabel('pixels')
plt.xlim(0,256)
plt.plot(dst_gray_hist)
plt.savefig('gray_hist.jpg')
plt.show()

#灰度图像均衡化前后对比
cv2.imshow('gray_grayHistEqualization',np.hstack([gray,dst_gray_histEqualization]))
cv2.imwrite('gray_grayHistEqualization.jpg',np.hstack([gray,dst_gray_histEqualization]))
cv2.waitKey()

#彩色图像直方图和均衡化
chans = cv2.split(img)
colors = ('b','g','r')
plt.figure(2)
plt.title('color_hist')
plt.xlabel('bins')
plt.ylabel('pixels')
plt.xlim(0,256)

for chan,color in zip(chans,colors):
    color_hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(color_hist,color = color)
plt.savefig('color_hist.jpg')
plt.show()

bHistE = cv2.equalizeHist(chans[0])
gHistE = cv2.equalizeHist((chans[1]))
rHistE = cv2.equalizeHist((chans[2]))
HistE = cv2.merge((bHistE,gHistE,rHistE))

cv2.imshow('color_colorHistE',np.hstack([img,HistE]))
cv2.imwrite('color_colorHistE.jpg',np.hstack([img,HistE]))
cv2.waitKey()