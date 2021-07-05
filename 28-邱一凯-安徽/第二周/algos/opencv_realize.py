#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project ：第二周 
@File    ：opencv_realize.py
@Author  ：Autumn
@Date    ：2021/6/30 19:48 
"""
from . import *


def cv_fun(src_img_path='test/lenna.png',
           dst_size=(800, 800, 3)):
    # openCV提供内置函数
    src_img = cv2.imread(src_img_path)
    # scale = (int(src_img.shape[1] / dst_size[1]), int(src_img.shape[0] / dst_size[0]))
    scale = (dst_size[1], dst_size[0])
    img_bilinear_interp = cv2.resize(src_img, scale, interpolation=cv2.INTER_LINEAR)
    img_nearest_interp = cv2.resize(src_img, scale, interpolation=cv2.INTER_NEAREST)
    img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    dst_img_name = "opencv_realize"
    # 水平组合
    img_h_stack = np.hstack([img_bilinear_interp, img_nearest_interp])
    # cv2.imshow(dst_img_name, img_gray)
    cv2.imshow(dst_img_name, img_h_stack)
    cv2.waitKey(0)
    return True
