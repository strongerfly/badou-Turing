#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：canny_manual.py
@Author  ：luigi
@Date    ：2021/7/13 下午3:48 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from 第三周 import convolution_manual

def canny(gray, ksize, sigma,lower,upper):
    """ canny算法实现，分4步：
        step#1：将灰度图高斯平滑
        step#2：通过sobel进行边缘提取
        step#3：非极大值抑制
        step#4：双阈值检测和链接边缘

    :param gray: 要检测边缘的灰度图
    :type gray: np.ndarray(np.uint8)
    :param ksize: 高斯核的size
    :type ksize: int
    :param sigma: 高斯核的sigma，决定了高斯分布的形态
    :type sigma: float
    :param lower: 双阈值的低阈值
    :type lower: int
    :param upper: 双阈值的低阈值
    :type upper: int
    :return:
    :rtype:
    """

    # 高斯平滑
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.dot(gaussian_1d, gaussian_1d.transpose())
    blur = convolution_manual.convolute(gray, gaussian_2d, padding=2, mode="same")
    # blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    
    # 计算图像梯度模与方向
    sobel_x = np.array(((-1,0,1),(-2,0,2),(-1,0,1)))
    sobel_y = np.array(((-1,-2,-1),(0,0,0),(1,2,1)))
    gradient_x = convolution_manual.convolute(blur, sobel_x, padding=1, mode="same")
    gradient_y = convolution_manual.convolute(blur, sobel_y, padding=1, mode="same")
    # gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = np.hypot(gradient_x,gradient_y)
    # gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y**2))
    gradient_x[gradient_x == 0] = 0.00000001
    gradient_theta_tan = gradient_y/gradient_x

    # 非极大值抑制NMS
    gradientWithBorder = cv2.copyMakeBorder(gradient, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    # case1: 0°< theta <=45°
    mask1 = (gradient_theta_tan>0) & (gradient_theta_tan<=1)
    coordinate1 = np.where(mask1)
    x,y = coordinate1
    weight1 = gradient_theta_tan[x,y]
    dTempA = gradientWithBorder[x-1, y+1]*weight1 + gradientWithBorder[x, y+1]*(1-weight1)
    dTempB = gradientWithBorder[x, y-1]*weight1 + gradientWithBorder[x+1, y-1]*(1-weight1)
    dTempMax1 = np.where(dTempA>dTempB, dTempA, dTempB)

    # case2: 45°< theta <=90°
    mask2 = (gradient_theta_tan>1)
    coordinate2 = np.where(mask2)
    x,y = coordinate2
    weight2 = gradient_theta_tan[x,y]
    dTempA = gradientWithBorder[x-1, y]*weight2 + gradientWithBorder[x-1, y+1]*(1-weight2)
    dTempB = gradientWithBorder[x+1, y-1]*weight2 + gradientWithBorder[x+1, y]*(1-weight2)
    dTempMax2 = np.where(dTempA>dTempB, dTempA, dTempB)

    # case3: 90°< theta <=135°
    mask3 = (gradient_theta_tan<=-1)
    coordinate3 = np.where(mask3)
    x,y = coordinate3
    weight3 = gradient_theta_tan[x,y]*(-1)  #weight need to be postive value
    dTempA = gradientWithBorder[x-1, y-1]*weight3 + gradientWithBorder[x-1, y]*(1-weight3)
    dTempB = gradientWithBorder[x+1, y]*weight3 + gradientWithBorder[x+1, y+1]*(1-weight3)
    dTempMax3 = np.where(dTempA>dTempB, dTempA, dTempB)

    # case4: 135°< theta <=180°
    mask4 = (gradient_theta_tan>-1) & (gradient_theta_tan<=0)
    coordinate4 = np.where(mask4)
    x,y = coordinate4
    weight4 = gradient_theta_tan[x,y]*(-1) #weight need to be postive value
    dTempA = gradientWithBorder[x-1, y-1]*weight4 + gradientWithBorder[x, y-1]*(1-weight4)
    dTempB = gradientWithBorder[x, y+1]*weight4 + gradientWithBorder[x+1, y+1]*(1-weight4)
    dTempMax4 = np.where(dTempA>dTempB, dTempA, dTempB)

    # combine all 4 coordinates together
    coordinateX = np.concatenate((coordinate1[0], coordinate2[0], coordinate3[0], coordinate4[0]))
    coordinateY = np.concatenate((coordinate1[1], coordinate2[1], coordinate3[1], coordinate4[1]))

    # compare each gradient point with the max sub-pixel in gradient direction
    candicate = gradient[coordinateX, coordinateY]
    dTempMax = np.concatenate((dTempMax1,dTempMax2,dTempMax3,dTempMax4))
    fake_gradient_exclude = np.where(candicate < dTempMax, 0, candicate)

    nms = gradient.copy()
    nms[coordinateX, coordinateY] = fake_gradient_exclude
    nms = nms[np.arange(gray.shape[0])].astype(np.uint8)

    # 双阈值检测
    doubleThreshold = np.where(nms < lower, 0, np.where(nms > upper, 255, nms))
    maskMid = (nms > lower) & (nms < upper)
    coordinate = np.where(maskMid)
    x,y = coordinate
    doubleThresholdWithBorder = cv2.copyMakeBorder(doubleThreshold, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    maskMax = doubleThresholdWithBorder[x,y]
    for i in range(-1,2):
        for j in range(-1,2):
            if not ((i==0) & (j==0)):
                maskMax = np.maximum(doubleThresholdWithBorder[x+i,y+i],maskMax)

    midThreshold = np.where((doubleThreshold[x,y]<maskMax), 0, 255)
    doubleThreshold[x,y] = midThreshold
    canny_manual = doubleThreshold[np.arange(gray.shape[0])]

    #cv2提供的接口（用于输出对比）
    canny_cv = cv2.Canny(blur.astype(np.uint8), lower, upper, apertureSize=3, L2gradient=True)
    # candy_cv = cv2.Canny(blur.astype(np.uint8), 0, 300)

    plt.subplot(221)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("gray")
    plt.imshow(gray, cmap='gray')
    plt.subplot(222)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("blur")
    plt.imshow(blur, cmap='gray')
    plt.subplot(223)
    plt.title("canny_manual")
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.imshow(canny_manual, cmap='gray')
    plt.subplot(224)
    plt.title("canny_cv")
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.imshow(canny_cv, cmap='gray')
    plt.show()

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--path",required=True, help="path for input image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ksize=5
    sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    m = np.median(gray)
    weight = 0.33
    lower = int(max(0, (1.0-weight) * m))
    upper = int(min(255, (1.0+weight) * m))

    canny(gray,ksize,sigma,lower,upper)

if __name__ == '__main__':
    main()
