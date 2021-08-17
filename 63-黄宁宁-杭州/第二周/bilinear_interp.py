#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def bilinear_interpolation(img,out_size):
    src_h,src_w,src_c=img.shape
    dst_h,dst_w=out_size[0],out_size[1]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x,scale_y=float(src_w) / dst_w, float(src_h) / dst_h
    for k in range(3):
        for i in range(dst_h):
            for j in range(dst_w):
                src_x = (j + 0.5) * scale_x - 0.5
                src_y = (i + 0.5) * scale_y - 0.5

                # 找到四个对应点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算两个过渡插值点后再计算最后的插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, k] + (src_x - src_x0) * img[src_y0, src_x1, k]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, k] + (src_x - src_x0) * img[src_y1, src_x1, k]
                dst_img[i, j, k] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img






if __name__=='__main__':
    img = cv2.imread('lenna.png')
    zoom= bilinear_interpolation(img, (800, 800))
    cv2.imshow('bilinear interp', zoom)
    cv2.waitKey()