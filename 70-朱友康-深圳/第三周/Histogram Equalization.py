#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

def equalizeHistDemo(dealGray):
    img = cv2.imread("../images/lenna.png", 1)
    if dealGray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 灰度图像直方图均衡化
        dst = cv2.equalizeHist(gray)

        # 直方图
        hist = cv2.calcHist([dst], [0], None, [256], [0, 256])#返回256灰度统计数值

        plt.figure()
        plt.hist(dst.ravel(), 256)
        plt.show()

        cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
        cv2.waitKey(0)
    else:
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        # 合并每一个通道
        result = cv2.merge((bH, gH, rH))
        cv2.imshow("dst_rgb", np.hstack([img, result]))
        cv2.waitKey(0)

if __name__ == "__main__":
    equalizeHistDemo(dealGray=True)