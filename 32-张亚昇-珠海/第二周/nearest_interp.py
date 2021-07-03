import numpy as np
import cv2
#读取图像
img = cv2.imread("D:/GoogleDownload/lenna.png")
cv2.imshow('img', img)
#创建空白图像
inter_img = np.zeros((800, 800, 3), dtype=np.uint8)
height_s = 800 / img.shape[0]
width_s = 800 / img.shape[1]
for i in range(800):
    for j in range(800):
        x = int(i / height_s)
        y = int(j / width_s)
        inter_img[i, j] = img[x, y]
#图像显示
cv2.imshow('inter_img', inter_img)
cv2.waitKey(0)


