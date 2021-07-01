# -*- coding: utf-8 -*-

import numpy as np
import cv2

def bilinear_interpolation(img,multiple):
    '''param
    img:输入图像;   multiple:放缩比例
    '''

    src_w, src_h, ch = img.shape  #原图像尺寸
    print(img.shape)
    # 目标图像尺寸
    dst_w = src_w * multiple
    dst_h = src_h * multiple

    output_img = np.zeros((dst_w, dst_h, 3))  # 目标图像初始化矩阵
    for i in range(ch):  # 通道遍历
        for dst_x in range(dst_w):  # 横轴遍历
            for dst_y in range(dst_h):   #众轴遍历


                # 计算目标图像对应于原图像点的坐标
                src_x = (dst_x + 0.5) * src_w / dst_w - 0.5
                src_y = (dst_y + 0.5) * src_h / dst_h - 0.5

                #原图像对应点周围邻近四个点的坐标
                x1 = int(src_x) ;  y1 = int(src_y)
                u = src_x - x1 ;   v = src_y -y1
                x2 = x1 + 1 ;      y2 = y1
                x3 = x1 ;          y3 = y1 +1
                x4 = x1 + 1 ;      y4 = y1 + 1

                #防止越界
                if x4 >= src_w:
                    x4 = src_w - 1
                    x2 = x4
                    x1 = x4 -1
                    x3 = x4 -1
                if y4 >= src_h:
                    y4 = src_h -1
                    y3 = y4
                    y1 = y4 -1
                    y2 = y4 -1

                #输出图像
                output_img[dst_x,dst_y,i] = int((1 - u) * (1 - v) * img[x1, y1, i] + \
                     u * (1 - v) * img[x2, y2, i ] + (1 - u) * v * img[x3, y3, i] + \
                     u * v * img[x4, y4, i])

    return output_img


if __name__ == '__main__':
    img = cv2.imread('E:/lenna.png').astype(np.float)
    out_image = bilinear_interpolation(img,2).astype(np.uint8)
    cv2.imshow('binary_linear_interpolation',out_image)
    cv2.imwrite('out_bilinear.jpg',out_image)
    cv2.waitKey()
