# -- coding: utf-8 --
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            #目标图像坐标点与原图像之间的转换
            x=int(i/sh)
            y=int(j/sw)
            #目标图像每个通道的二维矩阵点与原图像1个像素误差之内的像素点值相等
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("lenna.png")
zoom=function(img)
print(img)
print(img.shape)
print("-"*100)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)