import cv2
import numpy as np
from matplotlib import pyplot as plt


def  equalizeHist(img):
    h,w=img.shape[:2]
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    dst = np.zeros((h, w), np.uint8)
    copy=img.copy()
    init=0
    for i in range(256):
        init+=hist[i][0]
        x_index,y_index=np.where(copy==i)
        v=np.round(np.round(init / (h * w), 2)*256)-1
        if v<0:
            v=0
        dst[x_index,y_index]=v
    return dst

img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
dst=equalizeHist(gray)
# # 直方图
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)