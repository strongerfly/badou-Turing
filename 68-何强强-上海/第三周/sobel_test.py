# -*- coding:utf-8 -*-
"""
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
前四个是必须的参数：
src: 需要处理的图像；
ddepth: 图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
dx == 1:
    [[ 1,  0, -1],
    [ 2,  0, -2],
    [ 1,  0, -1]]
dy == 1:
    [[ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]]
其后是可选的参数：
dst: 目标图像；
ksize: Sobel算子的大小，必须为1、3、5、7。
scale: 缩放导数的比例常数，默认情况下没有伸缩系数；
delta: 一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType: 判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
"""
import cv2


if __name__ == '__main__':
    img = cv2.imread("tt.jpg", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_img_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
    sobel_img_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3)
    cv2.imshow("sobel_img_x", sobel_img_x)
    cv2.imshow("sobel_img_y", sobel_img_y)
    cv2.waitKey()