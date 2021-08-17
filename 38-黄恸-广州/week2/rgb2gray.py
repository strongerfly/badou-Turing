# -*- coding: utf-8 -*-

import cv2

'''
彩色图像灰度化
'''
img = cv2.imread("E:/lenna.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow('gary',img)
cv2.imwrite('gary.jpg', img)
cv2.waitKey()



'''
彩色图像二值化:
使用阈值（threshold）函数，将RGB图像转为二值图。
cv2.threshold(src, x, y, Methods)
src:指原图像，该原图像为灰度图
x:指用来对像素值进行分类的阈值
y:指当像素值高于（有时小于）阈值时应该被赋予的新的像素值
Methods：指不同的阈值方法，这些方法包括：cv2.THRESH_BINARY、cv2.THRESH_BINARY_INV、
cv2.THRESH_TRUNC、 cv2.THRESH_TOZERO、 cv2.THRESH_TOZERO_INV。
'''
# img1 = cv2.imread("E:/lenna.png")
# Grayimg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(Grayimg, 125, 255,cv2.THRESH_BINARY)
# cv2.imshow('binary', thresh)
# cv2.imwrite('binary.jpg', thresh)
# cv2.waitKey()

