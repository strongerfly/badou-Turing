# -*- coding:utf-8 -*-
"""
Canny边缘检测：优化的程序
cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
image: 原始图像
threshold1, threshold2：
    1. X < threshold1 丢弃X
    2. threshold1 < X < threshold2 弱边缘像素，观察其邻域像素（8个），若有一个为强像素则保留，否则丢弃
    3. X > threshold2 强边缘像素，保留
edges:
apertureSize: sobel 算子的大小  （3 * 3， 5 * 5， 7 * 7）
L2gradient: 布尔值，如果为真，则使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），否则使用L1范数（直接将两个方向导数的绝对值相加）

cv2.createTrackbar(trackbarName, windowName, value, count, onChange) 创建滑块 共有5个参数
    trackbarName，是这个trackbar对象的名字
    windowName，是这个trackbar对象所在面板的名字
    value，是这个trackbar的默认值,也是调节的对象
    count，是这个trackbar上调节的范围(0~count)
    onChange，是调节trackbar时调用的回调函数名, 回调函数需要一个参数来接收value(param3)

cv2.getTrackbarPos(trackbarName, windowName)
    当一个图像中有多个 trackBar时，可以通过cv2.getTrackbarPos(trackbarName, windowName) 获取指定滑块的值
"""
import cv2


low_threshold = 0
max_low_threshold = 500
kernel_size = 3
img, gray = None, None


def canny_threshold(pos):
    global img, gray
    t1 = cv2.getTrackbarPos("Min", "canny demo")
    t2 = cv2.getTrackbarPos("Max", "canny demo")
    print(t1, t2, pos)
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    detected_edges = cv2.Canny(detected_edges, t1, t2, apertureSize=kernel_size)  # 边缘检测
    # 显示找到的边缘
    cv2.imshow("detected_edges", detected_edges)

    # 用原始颜色添加到检测的边缘上， 之前过滤掉的像素点再加回来
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny demo', dst)


def test():
    global img, gray
    img = cv2.imread('tt.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
    cv2.namedWindow('canny demo')
    # 设置调节杠
    cv2.createTrackbar('Min', 'canny demo', low_threshold, max_low_threshold, canny_threshold)
    cv2.createTrackbar('Max', 'canny demo', max_low_threshold, max_low_threshold, canny_threshold)
    # 初始化
    canny_threshold(0)  # initialization
    if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
