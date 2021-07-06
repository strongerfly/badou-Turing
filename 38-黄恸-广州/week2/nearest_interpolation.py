# -*- coding: utf-8 -*-

import numpy as np
import cv2

def nearest_interpolation(img,multiple):
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
                src_x = round(dst_x* src_w / dst_w)
                src_y = round(dst_y * src_h / dst_h)

                #防止越界
                if src_x >= src_w:
                    src_x = src_w - 1

                if src_y >= src_h:
                    src_y = src_h -1

                #输出图像
                output_img[dst_x,dst_y,i] = img[src_x , src_y, i]

    return output_img



if __name__ == '__main__':
    img = cv2.imread('E:/lenna.png').astype(np.float)
    out_image = nearest_interpolation(img,2).astype(np.uint8)
    cv2.imshow('nearest_interpolation',out_image)
    cv2.imwrite('out_nearest.jpg',out_image)
    cv2.waitKey()
