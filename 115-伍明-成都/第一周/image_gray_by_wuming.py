# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:45:25 2021

@author: wuming
"""
import cv2
import numpy as np
def image_gray(img):#图像灰度化
    if img.ndim==2:
        print("请输入彩色图像")
        return 0
    height,width=img.shape[:2]
    img_gray=np.zeros([height,width],img.dtype)
    for i in range(height):
        for j in range(width):
            img_temp=img[i,j]
            img_gray[i,j]=int(img_temp[0]*0.11+img_temp[1]*0.59+img_temp[2]*0.3)#bgr格式
    return img_gray


image=cv2.imread("lenna.png")
gray_image=image_gray(image)
cv2.imshow("gray",gray_image)
#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#gray_image=image_gray(gray)
#cv2.imshow("gray",gray_image)
cv2.waitKey(0)
            