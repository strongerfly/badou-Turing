"""
@author:Suhao

彩色图像的临近插值
"""
import cv2
import numpy as np

#临近插值函数12345678


def function(img):
    height, width, channels = img.shape
    # 800*800
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    # 比例系数
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            # 取整数的过程，就是往左往右取舍的过程
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage
# 获取原来图像数据
img = cv2.imread("lenna.png")
# 进行临近插值
zoom = function(img)
# 打印数据和尺寸
print(zoom)
print(zoom.shape)
# 展示
cv2.imshow("nearest interp",zoom)
cv2.imshow("image", img)
# 以防计算过慢
cv2.waitKey(0)

