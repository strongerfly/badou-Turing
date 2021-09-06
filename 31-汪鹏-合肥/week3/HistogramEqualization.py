# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:59:02 2021

@author: wp
"""

import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

start =time.time()
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("Lenna_gray",gray)
# cv2.waitKey(10)


# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
# hist2 = cv2.calcHist([dst],[0],None,[256],[0,256])


plt.figure()
plt.hist(dst.ravel(),256)
# plt.subplot(121)
plt.plot(hist1)
plt.xlim([0,256])
print("Time:",time.time()-start)
# plt.subplot(122)
# plt.plot(hist2)
# plt.xlim([0,256])
# plt.show()

cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(10)


'''
# # 彩色图像直方图均衡化
b,g,r = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
res = cv2.merge([bH,gH,rH])
# cv2.imshow("Histogram Equalization",np.hstack([img,res]))
# cv2.waitKey(0)

chans = cv2.split(img)
chans2 = cv2.split(res)
colors = ('b', 'g', 'r')
# colors2 = ('o', 'y', 'p')
plt.figure()
plt.title('Flattened Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

for chan, color in zip(chans, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])
plt.show()
for chan, color in zip(chans2, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])
plt.show()
'''