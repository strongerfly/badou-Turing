# 1、最邻近插值的实现 2、双线性插值的实现 3、rgb2gray
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 灰度图
img = cv2.imread('lenna.png')
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        tmp = img[i, j]
        img_gray[i, j] = int(0.11*tmp[0]+0.59*tmp[1]+0.3*tmp[2]) # 灰度图一般都是0-255之间的整数，用于显示则需要转换

# 调用cv函数灰度化
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
def mybinary(img_gray):
    img_binary = np.where(img_gray >= 127, 255, 0)
    return img_binary

img_binary = mybinary(img_gray)
# cv2.imshow('binary', img_binary)
# cv2.waitKey(0)



# 最邻近插值法:新的图中的像素点映射进原图中与原图中哪个像素更靠近就填充为那个像素
def nearest(newx,newy,img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    newimg = np.zeros([newx, newy, channels], img.dtype)
    for i in range(newx):
        for j in range(newy):
            tmpx = int(i/(newx/height))
            tmpy = int(j/(newy/width))
            newimg[i, j] = img[tmpx, tmpy]
    return newimg


nearest_img = nearest(800, 800, 'lenna.png')


import math
def BiLinear_interpolation(img,dstH,dstW):
    scrH, scrW, _= img.shape
    img = np.pad(img, ((0, 1), (0, 1), (0, 0)), 'constant')
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = (i+1)*(scrH/dstH)-1
            scry = (j+1)*(scrW/dstW)-1
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx-x
            v = scry-y
            retimg[i, j] = (1-u)*(1-v)*img[x, y]+u*(1-v)*img[x+1, y]+(1-u)*v*img[x, y+1]+u*v*img[x+1, y+1]
    return retimg


