# -- coding: utf-8 --
import numpy as np
import cv2


def bilinear(img, out_dim):
    # 图像shape的返回值是（h,w,c）
    src_h, src_w, channel = img.shape
    # 目标图像尺寸
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 若目标图像的尺寸等于原图像则返回原图像的拷贝
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    #以dtype=np.uint8，创造一个目标图像的三维矩阵shape且值全部为0
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    #缩放比
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    #目标图像的三维矩阵，三个通道循环三次
    for i in range(3):
        #遍历每个通道上的二维矩阵
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 假设源图像与目标图像中心重合，两者坐标点之间的转换关系
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # np.floor()返回不大于输入参数的最大整数。（向下取整），这里是在原图像中找到两个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                #给当前目标图像位置像素点赋值
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear(img, (700, 700))
    cv2.imshow('origin', img)
    cv2.imshow('bilinear', dst)
    cv2.waitKey()
