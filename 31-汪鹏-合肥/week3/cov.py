# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 23:31:21 2021
@author: wp

实现最简单的卷积
bugs: 图像锐化+padding gray无法显示？？？
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

def convolution_one(inputs, filters, strides = 1, p_w = 0):  
    
    inputs = np.pad(inputs, pad_width=p_w, mode='constant', constant_values=0)
    h, w = inputs.shape    
    f = filters.shape[0]
      
    h_out = int((h - f) / strides) + 1
    w_out = int((w - f) / strides) + 1
    img_out = np.zeros([h_out, w_out])
    for i in range(h_out):
        for j in range(w_out):
            img_out[i, j] = np.sum(inputs[i * strides : i *strides + f, 
                            j * strides : j *strides + f] * filters)
    img_out = np.clip(img_out, 0, 255)   #将范围外的值变为255
    return img_out

#灰度图像卷积实现    
if __name__ == '__main__':
    img = cv2.imread('lenna.png', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # gy = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    ruihua = np.array([[-0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    stride = 1
    # p_w = 0
    p_w = int((ruihua.shape[0] - 1) / 2)
    # img_out_x = convolution_one(gray, gx, stride, p_w)
    # img_out_y = convolution_one(gray, gy, stride, p_w)
    img_out_ruihua = convolution_one(gray, ruihua, stride, p_w)
    inside = np.ones([img_out_ruihua.shape[0], 20]) * 255
    # cv2.imshow('lenna x,y', np.hstack([img_out_x, inside, img_out_y]))
    cv2.imshow('lenna org', gray)
    cv2.imshow('lenna ruihua', img_out_ruihua)
    # res = np.hstack([gray,inside, img_out_ruihua]) ######################
    # cv2.imshow('lenna ruihua', res)
    cv2.waitKey(0)

# # 彩色图像卷积
# if __name__ == '__main__':
#     img = cv2.imread('lenna.png', 1)
#     (b,g,r) = cv2.split(img)
#     filter_ = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
#     stride = 1
    
#     bh = convolution_one(b, filter_, stride)
#     gh = convolution_one(g, filter_, stride)
#     rh = convolution_one(r, filter_, stride)
#     result = cv2.merge((bh, gh, rh))
#     cv2.imshow('color lenna cov', img)
#     cv2.waitKey(0)
#     cv2.imshow('color lenna cov', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


               