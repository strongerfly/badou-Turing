# 第三周作业 1.直方图均衡化
# todo 公式实现
import cv2
import numpy
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equal = cv2.equalizeHist(gray)
cv2.imshow('gray and equal', numpy.hstack([gray, equal]))
cv2.waitKey()

plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

plt.figure()
plt.hist(equal.ravel(), 256)
plt.show()