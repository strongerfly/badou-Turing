#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：bilinear.py
@Author  ：luigi
@Date    ：2021/6/30 下午3:31 
'''

import numpy as np
import cv2

def bilinear_sample(input, x, y):
    """ 双线性插值法的实现

    :param input: 输入图像
    :type input: np.array(np.uint8)
    :param x: 目标图像矩阵在源图像矩阵中映射的行坐标(虚拟坐标)
    :type x: np.array(np.float64)
    :param y: 目标图像矩阵在源图像矩阵中映射的列坐标(虚拟坐标)
    :type y: np.array(np.float64)
    :return: 插值采样后的目标图像
    :rtype: np.array(np.uint8)
    """

    # 获取包含虚拟坐标的真实坐标：x1,y1,x2,y2
    x1 = x.astype(np.int32)
    y1 = y.astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1
    # 防止index xxx is out of bounds,将原图像填充边界，边界值等于最外层的值
    image_with_border = cv2.copyMakeBorder(input, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # 矩阵的行表示的是坐标系的列，矩阵的列表示的是坐标系的行
    # 如果下面公式中用所用图像矩阵image_with_border的index先用x后用y，则最后生成的图像是逆向旋转90度的
    z = (y2 - y) * (x2 - x) * image_with_border[y1, x1] \
        + (y2 - y) * (x - x1) * image_with_border[y1, x2] \
        + (y - y1) * (x2 - x) * image_with_border[y2, x1] \
        + (y - y1) * (x - x1) * image_with_border[y2, x2]

    return z.astype(np.uint8)