import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread("")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#灰度图均衡化
dst=cv2.equalizeHist(gray)
hist=cv2.calcHist([dst],[0],None,[256],[0,256])#计算直方图函数
hist.shape
plt.figure()
plt.show()

#rgb均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
