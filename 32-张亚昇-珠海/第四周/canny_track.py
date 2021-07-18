#参考源代码，实现修改canny阈值
import cv2
import numpy as np

def CannyThreshold(lowthreshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gaussian_img = cv2.GaussianBlur(gray, (3, 3), 0) #高斯滤波
    detected_edgs = cv2.Canny(Gaussian_img, lowthreshold, lowthreshold*ratio,
                              apertureSize=kernel_size) #边缘检测
    dst = cv2.bitwise_and(img, img, mask=detected_edgs)
    cv2.imshow('canny demo', dst)

lowthreshold = 0
max_lowthreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread("D:/GoogleDownload/lenna.png")

cv2.namedWindow('canny demo')
cv2.createTrackbar('Min threshold', 'canny demo', lowthreshold, max_lowthreshold, CannyThreshold)

CannyThreshold(0)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()