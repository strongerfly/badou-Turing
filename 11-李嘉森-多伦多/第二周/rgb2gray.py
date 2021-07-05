import numpy as np
import cv2


def rgb2gray(src_img):
    src_y_dim, src_x_dim, src_d_dim = src_img.shape
    dst_img = np.zeros((src_y_dim, src_x_dim), dtype=np.uint8)

    for y in range(src_y_dim):
        for x in range(src_x_dim):
            dst = 0.3 * src_img[y, x, 2] + 0.59 * src_img[y, x, 1] + 0.11 * src_img[y, x, 0]
            dst_img[y, x] = dst

    return dst_img


if __name__ == '__main__':
    img_in = cv2.imread('lenna.png')
    img_out = rgb2gray(img_in)
    print(img_out.shape)
    cv2.imshow('rgb2gray', img_out)
    cv2.waitKey()
