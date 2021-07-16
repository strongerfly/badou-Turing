# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:23:53 2021

@author: wp
"""

import cv2
import numpy as np

# img = cv2.imread("lenna.png")

# h, w, c = img.shape
# h1, w1 = 200,200
# sh = h / h1
# sw = w / w1

# img_resize = np.zeros([h1, w1, c], np.uint8)

# # for i1 in range(w1):
# #     for j1 in range(h1):
# #         i = int(i1 * sw)
# #         j = int(j1 * sh)
# #         im = min(i+1, w-1)
# #         jm = min(j+1, h-1)
        
# #         x0 = i1 * sw - i
# #         y0 = j1 * sh - j

# #         img_resize[i1, j1] = (1- y0)*((1 - x0)* img[i, j]+ x0 *img[im, j]) + y0*((1 - x0)* img[i, jm]+ x0 *img[im, jm])

# for i1 in range(w1):  # 新图像与原图像中心点对齐
#     for j1 in range(h1):
#         i = int((i1 + 0.5) * sw - 0.5)
#         j = int((j1 + 0.5) * sh - 0.5)
#         im = min(i+1, w-1)
#         jm = min(j+1, h-1)

#         x0 = i1 * sw - i
#         y0 = j1 * sh - j
    
#         img_resize[i1, j1]  = (1- y0)*((1 - x0)* img[i, j]+ x0 *img[im, j]) + y0*((1 - x0)* img[i, jm]+ x0 *img[im, jm])

            
# # cv2.imshow("img",img)
# cv2.imshow("img_resize",img_resize)
# cv2.waitKey(10)

def bilinear_interpolation(img, resize_shape):
    src_h, src_w, channel = img.shape
    dst_w, dst_h = resize_shape[:2]  ####!!!
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h ==dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype = np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for c in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x + 0.5
                src_y = (dst_y + 0.5) * scale_y + 0.5
                
                src_x0 = int(src_x)
                src_y0 = int(src_y)
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)
                
                tmpx = (src_x1 - src_x) * img[src_x0, src_y0, c] + (src_x - src_x0) * img[src_x1, src_y0, c]
                tmpy = (src_x1 - src_x) * img[src_x0, src_y1, c] + (src_x - src_x0) * img[src_x1, src_y1, c]
                
                dst_img[dst_x, dst_y,c ] = (src_y1 - src_y) * tmpx + (src_y - src_y0) * tmpy
    return dst_img


if __name__ == '__main__':
    img = cv2.imread('mm.jpg')
    dst = bilinear_interpolation(img, (520, 292))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()


























