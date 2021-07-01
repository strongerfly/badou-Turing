# -*- coding:utf-8 -*-
"""
imread函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式，有三种：

cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
cv2.IMREAD_UNCHANGED：包括alpha，可以直接写-1。
"""
import cv2


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    print(img.shape)
    cv2.imshow("origin", img)
    cv2.waitKey()
