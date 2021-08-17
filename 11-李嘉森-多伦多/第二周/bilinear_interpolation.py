import numpy as np
import cv2


def bilinear_interpolation(src_img, dst_dim):
    src_y_dim, src_x_dim, src_d_dim = src_img.shape
    dst_y_dim, dst_x_dim = dst_dim

    dst_img = np.zeros((dst_y_dim, dst_x_dim, src_d_dim), dtype=np.uint8)

    src_x_ratio = src_x_dim / dst_x_dim  # 新图缩放到原图的比例： x轴比例
    src_y_ratio = src_y_dim / dst_y_dim  # 新图缩放到原图的比例： y轴比例

    for dst_y in range(dst_y_dim):
        for dst_x in range(dst_x_dim):
            for i in range(src_d_dim):
                # 新图像素在原图的坐标
                src_x = (dst_x + 0.5) * src_x_ratio - 0.5  # x轴坐标
                src_y = (dst_y + 0.5) * src_y_ratio - 0.5  # y轴坐标

                # 找到该像素相邻的四个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_x_dim - 1)  # 如果 x0坐标+1超出图像宽度，那么x1坐标使用x轴边界的点

                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_y_dim - 1)  # 如果 y0坐标+1超出图像长度，那么y1坐标使用y轴边界的点

                # 根据公式在x轴相邻两点之间做插值
                f_r1 = (src_x1 - src_x) * src_img[src_y0, src_x0, i] + (src_x - src_x0) * src_img[src_y0, src_x1, i]
                f_r2 = (src_x1 - src_x) * src_img[src_y1, src_x0, i] + (src_x - src_x0) * src_img[src_y1, src_x1, i]

                # 根据公式在y轴相邻两点之间做插值
                f_xy = (src_y1 - src_y) * f_r1 + (src_y - src_y0) * f_r2  # 该像素在插值位置通道i的像素值

                dst_img[dst_y, dst_x, i] = int(f_xy)  # 将通道i的像素值存入新图像素对应的位置

    return dst_img


if __name__ == '__main__':
    img_in = cv2.imread('lenna.png')
    img_out = bilinear_interpolation(img_in, (700, 700))
    cv2.imshow('bilinear interpolation', img_out)
    cv2.waitKey()
