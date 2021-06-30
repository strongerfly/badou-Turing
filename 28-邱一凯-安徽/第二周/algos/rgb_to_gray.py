#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project ：第二周 
@File    ：rgb_to_gray.py
@Author  ：Autumn
@Date    ：2021/6/30 16:40 
"""
from . import *


# 浮点算法
def float_algo(r, g, b):
    return r * 0.3 + g * 0.59 + b * 0.11


# 整数算法
def int_algo(r, g, b):
    return (r * 30 + g * 59 + b * 11) / 100


# 移位算法
def shifting_algo(r, g, b):
    return (r * 76 + g * 151 + b * 28) >> 8


# 平均值法
def avg_algo(r, g, b):
    return (int(r) + int(g) + int(b)) / 3


# 仅取绿色
def green_only(r, g, b):
    return g


# 灰度化图片
def rgb2gray(src_img_path='test/lenna.png',
             new_save_path1='test/img_gray.png',
             new_save_path2='test/img_binary.png'):
    """
    :param src_img_path: this is old image path
    :param new_save_path2: this is new image path
    :param new_save_path1: this is new image path
    :return: this is new image matrix
    """
    src_img = cv2.imread(src_img_path)
    src_size = src_img.shape
    dst_img = np.zeros((src_size[0], src_size[1], 1), dtype=np.uint8)
    binary_dst_img = np.zeros((src_size[0], src_size[1], 1), dtype=np.uint8)

    fun_list = [float_algo, int_algo, shifting_algo, avg_algo, green_only]  # 灰度图转换函数列表
    dst_img_list = []

    for fun in fun_list:
        for src_y in range(src_size[0]):
            for src_x in range(src_size[1]):
                # openCV按BGR顺序读取三通道
                dst_img[src_y, src_x] = fun(r=src_img[src_y, src_x, 2],
                                            g=src_img[src_y, src_x, 1],
                                            b=src_img[src_y, src_x, 0])

                # 二值化
                if fun == float_algo:
                    if dst_img[src_y, src_x] <= 50:
                        binary_dst_img[src_y, src_x] = 0
                    else:
                        binary_dst_img[src_y, src_x] = 255
                    # img_binary = np.where(img_gray >= 0.5, 1, 0)
        dst_img_list.append(dst_img)
    dst_img_list.append(binary_dst_img)
    dst_img_name = "rgb2gray"
    # 水平组合
    img_h_stack = np.hstack(dst_img_list)
    cv2.imshow(dst_img_name, img_h_stack)

    cv2.imwrite(new_save_path1, dst_img)  # 保存图片
    cv2.imwrite(new_save_path2, binary_dst_img)

    cv2.waitKey(0)
    return dst_img
