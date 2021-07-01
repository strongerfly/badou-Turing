#    @author Created by Genius_Tian

#    @Date 2021/6/27

#    @Description AI学习第二周作业
import numpy as np
import cv2
import time


def bilinear_interpolation(src_img, target_shape):
    h, w, channel = src_img.shape
    dst_h, dst_w = target_shape[1], target_shape[0]
    if dst_h == h and dst_w == w:
        return src_img.copy()

    empty_img = np.zeros((dst_h, dst_w, channel), src_img.dtype)
    scale_x, scale_y = float(w) / dst_w, float(h) / dst_h
    for c in range(channel):
        for dst_y in range(dst_h):
            src_y = (dst_y + 0.5) * scale_y - 0.5
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5

                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))

                src_x1 = min(src_x0 + 1, w - 1)
                src_y1 = min(src_y0 + 1, h - 1)
                r1 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                r2 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                empty_img[dst_y, dst_x, c] = ((src_y1 - src_y) * r1 + (src_y - src_y0) * r2).astype(int)

    return empty_img


def my_bilinear_interpolation(src_img, target_shape):
    h, w, channel = src_img.shape
    dst_h, dst_w = target_shape
    if dst_h == h and dst_w == w:
        return src_img.copy()

    zero_x = float(w) / dst_w
    zero_y = float(h) / dst_h
    # 记录矩阵所有x索引和y索引
    x_index, y_index = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
    src_x = (x_index + 0.5) * zero_x - 0.5
    src_y = (y_index + 0.5) * zero_y - 0.5

    src_x0 = np.clip(np.floor(src_x).astype(int), 0, w - 1)
    src_y0 = np.clip(np.floor(src_y).astype(int), 0, h - 1)
    # src_x0 = np.floor(src_x).astype(int)
    # src_y0 = np.floor(src_y).astype(int)

    src_x1 = np.clip(src_x0 + 1, 0, w - 1)
    src_y1 = np.clip(src_y0 + 1, 0, h - 1)
    r1 = (src_x1 - src_x)[..., np.newaxis] * img[src_y0, src_x0, :] + (src_x - src_x0)[..., np.newaxis] * img[src_y0,
                                                                                                          src_x1, :]
    r2 = (src_x1 - src_x)[..., np.newaxis] * img[src_y1, src_x0, :] + (src_x - src_x0)[..., np.newaxis] * img[src_y1,
                                                                                                          src_x1, :]

    return ((src_y1 - src_y)[..., np.newaxis] * r1 + (src_y - src_y0)[..., np.newaxis] * r2).astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    start = time.time()
    t1 = bilinear_interpolation(img, (800, 800))
    end1 = time.time()
    t2 = my_bilinear_interpolation(img, (800, 800))
    end2 = time.time()
    print("循环耗时%.3f s,numpy向量加速耗时%.3f s" % (end1 - start, end2 - end1))
    # print(np.sum(t2 - t1))
    cv2.imshow("t2", t2)
    cv2.waitKey(0)
