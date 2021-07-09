# 彩色图像的灰度化、二值化
# 灰度化1
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img = cv2.imread('lenna.png')
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

cv2.imshow('img', img)
print(img_gray[1, 1])
cv2.imshow('img_gray', img_gray)
# 灰度化2
img_gray1 = rgb2gray(img)
cv2.imshow('img_gray1', img_gray1)
# 灰度化3
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img_gray2', img_gray2)
# 二值化1
img = plt.imread('lenna.png')  # plt.inread 读取出的是数值为0-1之间的一个矩阵
print(img)
img_gray = rgb2gray(img)
print(img_gray)
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if img_gray[i, j] <= 0.5:
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
# cv2.imshow('img_binary1', img_gray)
# 二值化2
img_binary = np.where(img_gray >= 0.5, 1, 0)#int32
img_binary1=np.array(img_binary*255,dtype='uint8')
cv2.imshow('img_binary1',img_binary1)
cv2.waitKey()
cv2.destroyAllWindows()



