import numpy as np
import cv2
#灰度化
#数据读入
img = cv2.imread("D:/GoogleDownload/lenna.png")
print('Image shape:', img.shape)
cv2.imshow('img', img)
print(img.dtype)
#创建空图像
new_img = np.zeros([img.shape[0],img.shape[1]], img.dtype)
print('new_img shape:', new_img.shape)
print(img[1][2][0])
print(img[1][2][1])
print(img[1][2][2])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_img[i,j] = img[i][j][0]*0.11 + img[i][j][1]*0.59 + img[i][j][2]*0.3


cv2.imshow("gray_img", new_img)
cv2.waitKey(0)
print('new_img shape:', new_img.shape)
cv2.imwrite("D:/GoogleDownload/lenna_gray.png", new_img)
#二值化
gray_img = new_img.copy()
print('gray_img shape:', gray_img.shape)
gray_img = gray_img / 255 #归一化
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if gray_img[i, j] <= 0.5:
            gray_img[i, j] = 0
        else:
            gray_img[i, j] = 1

print(gray_img)
cv2.imshow("binarization",gray_img)
