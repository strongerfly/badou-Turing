# -*- coding: utf-8 -*-
"""

@author: wuming
"""
#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
import matplotlib as plt
import math
def canny_oper(img,threshold1,threshold2):
    #构建高斯模板
    sigma=0.5
    dim=int(np.round(6*sigma+1))
    if dim %2==0:
        dim+=1
    Gaussian_filter=np.zeros([dim,dim])#用于存储高斯核
    temp=[i-dim//2 for i in range(dim)]#//取整除 - 返回商的整数部分,向下取整
    n1=1/(2*math.pi*sigma**2)
    n2=-1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i,j]=n1*math.exp(n2*(temp[i]**2+temp[j]**2))
    Gaussian_filter=Gaussian_filter/Gaussian_filter.sum()
    #高斯平滑
    print(img.shape)
    dx,dy=img.shape
    img_new=np.zeros(img.shape)
    temp=dim//2
    img_pad=np.pad(img,((temp,temp),(temp,temp)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i,j]=np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)#得到高斯滤波后的图像
    #灰度梯度
    sobel_kernel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    img_tidu_x=np.zeros(img.shape)
    img_tidu_y=np.zeros(img.shape)
    img_tidu=np.zeros(img_new.shape)
    img_pad=np.pad(img_new,((1,1),(1,1)))
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j]=np.sum(img_pad[i:i + 3, j:j + 3]*sobel_kernel_x)
            img_tidu_y[i,j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i,j]=np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
    img_tidu_x[img_tidu_x==0]=0.00000001
    angle=img_tidu_y/img_tidu_x

    #非极大值抑制
    img_yizhi=np.zeros(img.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag=True
            temp=img_tidu[i-1:i+2,j-1:j+2]
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                num_2 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[2, 0] - temp[2, 1]) * angle[i, j] + temp[2, 1]
                num_2 = (temp[0, 2] - temp[0, 1]) * angle[i, j] + temp[0, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[0, 1] - temp[0, 0]) * angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) * angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    #双边值检测，边缘连接
    # lower_boundary=img_tidu.mean()*0.5
    # high_boundary=img_tidu.mean()*3
    lower_boundary = threshold1
    high_boundary = threshold2
    zhan=[]
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            if img_yizhi[i,j]>=high_boundary:
                img_yizhi[i,j]=255
                zhan.append([i,j])
            elif img_yizhi[i,j]<=lower_boundary:
                img_yizhi[i,j]=0
    while not len(zhan)==0:
        temp_1,temp_2=zhan.pop()
        a=img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    cv2.imshow("canny",img_yizhi.astype(np.uint8))
    cv2.waitKey(0)

if __name__=='__main__':
    img=cv2.imread('lenna.png')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img",gray)
    cv2.waitKey(10)
    canny_oper(gray, 50, 100)




