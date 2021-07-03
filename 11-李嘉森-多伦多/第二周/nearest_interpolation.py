import numpy as np
import cv2


def nearest_interpolation(src_img, dst_dim):
    src_y_dim, src_x_dim, src_d_dim = src_img.shape
    dst_y_dim, dst_x_dim = dst_dim

    dst_img = np.zeros((dst_y_dim, dst_x_dim, src_d_dim), dtype=np.uint8)

    src_x_ratio = src_x_dim / dst_x_dim  # 新图缩放到原图的比例： x轴比例
    src_y_ratio = src_y_dim / dst_y_dim  # 新图缩放到原图的比例： y轴比例

    for dst_y in range(dst_y_dim):
        for dst_x in range(dst_x_dim):
            for i in range(src_d_dim):
                src_x = (dst_x + 0.5) * src_x_ratio - 0.5  # x轴坐标
                src_y = (dst_y + 0.5) * src_y_ratio - 0.5  # y轴坐标

                src_x_round = int(src_x)
                src_y_round = int(src_y)

                dst_img[dst_y, dst_x, i] = src_img[src_y_round, src_x_round, i]  # 将通道i的像素值存入新图像素对应的位置

    return dst_img


if __name__ == '__main__':
    img_in = cv2.imread('lenna.png')
    img_out = nearest_interpolation(img_in, (700, 700))
    cv2.imshow('nearest interpolation', img_out)
    cv2.waitKey()
