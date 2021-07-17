
import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    blured = cv2.GaussianBlur(gray,(3,3),0)
    edges =cv2.Canny(blured,lowThreshold,highThreshold,lowThreshold*ratio,apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=edges)  # 用原始颜色添加到检测的边缘上
    cv2.imshow("canny demo",edges)

img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("canny demo")
lowThreshold = 0
highThreshold = 100
ratio = 3
kernel_size = 3
cv2.createTrackbar("canny","canny demo",lowThreshold,highThreshold,CannyThreshold)
CannyThreshold(0)  # initialization

while(True):
    #等待关闭
    k=cv2.waitKey(0)&0xFF
    if k==27:
        break