import cv2
import numpy as np
minValue, maxValue = 0, 0  # 阈值

def showing(x):  # 回调函数更新图像
    # 获取滑条位置
    minValue = cv2.getTrackbarPos("minValue","canny")
    maxValue = cv2.getTrackbarPos("maxValue","canny")
    edges = cv2.Canny(img,minValue,maxValue)
    cv2.imshow('canny', edges)


img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png", 0)  # 读取图片
# cv2.namedWindow('image')  # 显示原图
# cv2.imshow('image', img)
cv2.namedWindow('canny')  # 显示检测图
# 创建两个滑条
cv2.createTrackbar('minValue', 'canny', 0, 255, showing)
cv2.createTrackbar('maxValue', 'canny', 0, 255, showing)

edges = cv2.Canny(img, minValue, maxValue)
cv2.imshow('canny', edges)
while(True):
    #等待关闭
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
cv2.destroyAllWindows()