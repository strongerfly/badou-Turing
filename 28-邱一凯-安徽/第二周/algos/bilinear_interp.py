#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project ：第二周 
@File    ：bilinear_interp.py
@Author  ：Autumn
@Date    ：2021/6/30 12:20 
"""
from . import *


# 单线性插值
def sig_linear_interp(co_x0, co_y0, co_x1, co_y1, co_src):
    args1 = (co_x1 - co_src) / (co_x1 - co_x0) * co_y0
    args2 = (co_src - co_x0) / (co_x1 - co_x0) * co_y1
    return int(args1 + args2)


# 双线性插值
def bilinear_interp(src_img_path='test/lenna.png',
                    dst_size=(700, 700, 3),
                    new_save_path='test/bilinear_interp_img.png'):
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

    scale_x = src_size[1] / dst_size[1]
    scale_y = src_size[0] / dst_size[0]

    for dst_y in range(dst_size[0]):
        for dst_x in range(dst_size[1]):
            src_x = (dst_x + 0.5) * scale_x - 0.5  # 宽度
            src_y = (dst_y + 0.5) * scale_y - 0.5  # 高度

            src_x0 = max(int(src_x), 0)  # int(np.floor(src_x))
            src_y0 = max(int(src_y), 0)  # int(np.floor(src_y))
            src_x1 = min(src_x0 + 1, src_size[0] - 1)
            src_y1 = min(src_y0 + 1, src_size[1] - 1)
            for shape_l in range(src_size[-1]):
                # 右边界的特殊处理
                if src_x0 == src_size[1] - 1:
                    dst_value = sig_linear_interp(co_x0=src_y0,
                                                  co_y0=src_img[src_y0, src_x0, shape_l],
                                                  co_x1=src_y1,
                                                  co_y1=src_img[src_y1, src_x0, shape_l],
                                                  co_src=src_y)
                # 下边界的特殊处理
                elif src_y0 == src_size[1] - 1:
                    dst_value = sig_linear_interp(co_x0=src_x0,
                                                  co_y0=src_img[src_y0, src_x0, shape_l],
                                                  co_x1=src_x1,
                                                  co_y1=src_img[src_y0, src_x1, shape_l],
                                                  co_src=src_x)
                else:
                    # 在X轴方向做双线性插值
                    temp_value1 = sig_linear_interp(co_x0=src_x0,
                                                    co_y0=src_img[src_y0, src_x0, shape_l],
                                                    co_x1=src_x1,
                                                    co_y1=src_img[src_y0, src_x1, shape_l],
                                                    co_src=src_x)
                    temp_value2 = sig_linear_interp(co_x0=src_x0,
                                                    co_y0=src_img[src_y1, src_x0, shape_l],
                                                    co_x1=src_x1,
                                                    co_y1=src_img[src_y1, src_x1, shape_l],
                                                    co_src=src_x)
                    # 在Y轴方向做双线性插值
                    dst_value = sig_linear_interp(co_x0=src_y0,
                                                  co_y0=temp_value1,
                                                  co_x1=src_y1,
                                                  co_y1=temp_value2,
                                                  co_src=src_y)
                dst_img[dst_y, dst_x, shape_l] = dst_value

    dst_img_name = "bilinear_interp"
    cv2.imshow(dst_img_name, dst_img)  # 显示图片

    cv2.imwrite(new_save_path, dst_img)  # 保存图片

    time_end = time.time()
    print("The time of bilinear interpolation is {:.2f} s".format(time_end - time_start))

    cv2.waitKey(0)
    return dst_img
