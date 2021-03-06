"""
@author:Suhao

彩色图像的双线性插值
"""
 
import numpy as np
import cv2
 
'''
python implementation of bilinear interpolation
'''
# 双线性插值函数12345678


def bilinear_interpolation(img,out_dim):
    # 原图像尺寸
    src_h, src_w, channel = img.shape
    # 目标图像尺寸
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 判断原图像和目标图像尺寸是否一致
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    # 创建一个空的图像
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 长宽比例系数
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                # 进行坐标中心重合（坐标系选择）
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                # 一次
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 二次
                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img
 
 
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (600, 600))
    cv2.imshow('bilinear_interp', dst)
    cv2.waitKey()
