# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 20:26:43 2021

@author: wp
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# # 灰度图像直方图
img = cv2.imread('lenna.png', 1)
# b, g, r = cv2.split(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


''' 
# Counter查找，待优化
ba= np.array(gray.ravel())
va = np.zeros([256,1], np.uint8)

for xs in range(256):
    va[xs] = Counter(ba)[xs]  # 此函数很慢
plt.figure()
plt.plot(va)
plt.show()
'''

''' 
# 直接显示-hist
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()
'''


hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title('Gray Lenna Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(hist)
plt.xlim([0,256])
plt.show()


'''
# 彩色图像直方图
img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# b,g,r = cv2.split(img)
# c = cv2.merge([b,g,r,gray])
# chans = cv2.split(c)    # 此处考虑下直接合并？
# colors = ('b', 'g', 'r','gray')
chans = cv2.split(img)    
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Flattened Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

for chan, color in zip(chans, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist, color = color)
    plt.xlim([0,256])
plt.show()
'''


