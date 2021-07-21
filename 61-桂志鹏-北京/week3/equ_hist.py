# 直方图均衡化实现
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../img/lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 获取灰度图
cv2.imshow('img', img)

gray_equ_hist = cv2.equalizeHist(gray)

# hist = cv2.calcHist([img], [2], None, [256], [0, 256])
# hist = cv2.calcHist([gray], 02], None, [256], [0, 256])

print(gray_equ_hist)
plt.hist(gray_equ_hist.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, gray_equ_hist]))

# 彩色图的直方图均衡化
(b, g, r) = cv2.split(img)

b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)

bgr_hist = cv2.merge((b_hist, g_hist, r_hist))
cv2.imshow("bgr_hist", bgr_hist)

cv2.waitKey(0)
cv2.destroyAllWindows()  # 关闭所有窗口
