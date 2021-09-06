# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lenna.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''
cv2 def Sobel(src: Any, 要处理的数据
          ddepth: Any,  图像的深度
          dx: Any,      x方向求导阶数
          dy: Any,      y方向求导阶数
          dst: Any = None, 结果数据
          ksize: Any = None,    kernel size sobel核的大小，一般为1，3，5，7
          scale: Any = None,    缩放导数的比例常数，默认情况下没有伸缩系数；
          delta: Any = None,    一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
          borderType: Any = None) -> None  判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''
gray_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gray_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

gray_laplace = cv2.Laplacian(gray, cv2.CV_64F,ksize=3)

gray_canny = cv2.Canny(gray, 100, 200)

plt.subplot(231), plt.imshow(gray, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(gray_sobel_x, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(gray_sobel_y, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(gray_laplace,  "gray"), plt.title("Laplace")
plt.subplot(235), plt.imshow(gray_canny, "gray"), plt.title("Canny")
plt.show()