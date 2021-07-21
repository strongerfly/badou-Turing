import math

import numpy as np
import cv2
from matplotlib import pyplot as plt
def hist_1():
    gray=cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
    #画直方图
    hist=cv2.calcHist([gray],[0],None,[256],[0,255])
    plt.plot(hist)
    plt.show()
    # 直方图均值化
    gray_hist=cv2.equalizeHist(gray)
    cv2.imshow("gray",gray)
    cv2.imshow("gray_hist",gray_hist)
    # cv2.waitKey(0)

    img=cv2.imread("lenna.png")
    (b,g,r)=cv2.split(img)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    color = cv2.merge((b,g,r))
    cv2.imshow("img", img)
    cv2.imshow("color", color)
    cv2.waitKey(0)
def hist_2():
    gray = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
    w,h=gray.shape[0],gray.shape[1]
    scale = 256/(w*h)
    min,max = np.min(gray),np.max(gray)
    hist = np.zeros_like(gray)
    for i in range(min,max+1):
        temp1 = np.where(gray < (i + 1))
        #像素值小于i+1的总像素点
        num = np.array(temp1).shape[1]
        q=math.ceil(num*scale-1)
        hist[np.where(gray==i)]=q

    cv2.imshow("hist_before",gray)
    cv2.imshow("hist_after",hist)
    cv2.waitKey(0)




if __name__=="__main__":
    hist_1()
    hist_2()