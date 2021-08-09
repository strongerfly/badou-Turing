
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('lenna_gray',gray)

##显示直方图
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()
#cv2.waitKey(0)

##直方图均衡
dst=cv2.equalizeHist(gray)
hist=cv2.calcHist(dst,[0],None,[256],[0,256])

plt.figure()
plt.title("qualizeHist")
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)