#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project ：第二周 
@File    ：nearest_interp.py
@Author  ：Autumn
@Date    ：2021/6/29 21:34 
"""

from . import *


# 最邻近插值
def nearest_interp(src_img_path='test/lenna.png',
                   dst_size=(800, 800, 3),
                   new_save_path='test/nearest_interp_img.png'):
    """
    :param src_img_path: this is old image path
    :param dst_size: this is new image size
    :param new_save_path: this is new image path
    :return: this is new image matrix
    """
    time_start = time.time()
    src_img = cv2.imread(src_img_path)
    src_size = src_img.shape
    dst_img = np.zeros(dst_size, dtype=np.uint8)

    for dst_y in range(dst_size[0]):
        for dst_x in range(dst_size[1]):
            src_x = dst_x * (src_size[1] / dst_size[1])  # 宽度
            src_y = dst_y * (src_size[0] / dst_size[0])  # 高度
            # for shape_l in range(src_size[-1]):
            dst_img[dst_y, dst_x] = src_img[int(src_y), int(src_x)]

    dst_img_name = "nearest_interp"
    cv2.imshow(dst_img_name, dst_img)  # 显示图片

    cv2.imwrite(new_save_path, dst_img)  # 保存图片

    time_end = time.time()
    print("The time of nearest interpolation is {:.2f} s".format(time_end - time_start))

    cv2.waitKey(0)
    return dst_img
