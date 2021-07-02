#    @author Created by Genius_Tian

#    @Date 2021/6/27

#    @Description AI学习第二周作业
import time

import cv2
import numpy as np

"""
原版最临近插值算法
"""


def nearest_interpolation(src_img, target_shape):
    th, tw = target_shape
    h, w, channel = src_img.shape
    if th == h and tw == w:
        return src_img.copy()
    empty_img = np.zeros((th, tw, channel), src_img.dtype)
    sh = th / h
    sw = tw / w
    scale_x, scale_y = float(w) / tw, float(h) / th
    for i in range(th):
        src_y = (i + 0.5) * scale_y - 0.5
        for j in range(tw):
            src_x = (j + 0.5) * scale_x - 0.5
            x = int(src_x / sh)
            y = int(src_y / sw)
            empty_img[i, j] = src_img[x, y]
    return empty_img


def my_nearest_interpolation(src_img, target_shape):
    th, tw = target_shape
    h, w, channel = src_img.shape
    if th == h and tw == w:
        return src_img.copy()
    x_index, y_index = np.meshgrid((np.arange(tw) * w / tw).astype(int), (np.arange(th) * h / th).astype(int))
    xy_index = np.concatenate((x_index[..., np.newaxis], y_index[..., np.newaxis]), axis=2)
    return img[xy_index[:, :, 1], xy_index[:, :, 0], :]


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    start = time.time()
    resize_img = nearest_interpolation(img, (800, 800))
    end1 = time.time()
    interpolation = my_nearest_interpolation(img, (800, 800))
    end2 = time.time()
    print("循环耗时%.3f s,numpy向量加速耗时%.3f s" % (end1 - start, end2 - end1))
    cv2.imshow("nearest", interpolation)
    cv2.waitKey(0)
