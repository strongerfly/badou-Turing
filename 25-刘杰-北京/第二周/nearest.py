#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：nearest.py
@Author  ：luigi
@Date    ：2021/6/30 下午3:16 
'''

import numpy as np


def nearest_sample(input, x, y):
    """ 最邻近插值法的实现

    :param input: 输入图像
    :type input: np.array(np.uint8)
    :param x: 目标图像矩阵在源图像矩阵中映射的行坐标(虚拟坐标)
    :type x: np.array(np.float64)
    :param y: 目标图像矩阵在源图像矩阵中映射的列坐标(虚拟坐标)
    :type y: np.array(np.float64)
    :return: 插值采样后的目标图像
    :rtype: np.array(np.uint8)
    """

    x1 = np.rint(x).astype(np.int32)
    y1 = np.rint(y).astype(np.int32)
    return input[y1, x1]
