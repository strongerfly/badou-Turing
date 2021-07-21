# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import  sys
from skimage.color import rgb2gray as skrgb2gray
from sklearn.datasets import load_iris
import sklearn.decomposition as dp


# 自己实现直方图均衡化
def histogram_equalization( img ):
    h, w, c = img.shape
    r = [0] * 256
    g = [0] * 256
    b = [0] * 256
    # 计算各像素点数量
    for x in range(h):
        for y in range(w):
            bi, gi, ri = img[x, y]
            b[bi] += 1
            g[gi] += 1
            r[ri] += 1
    img_h = np.zeros(img.shape, img.dtype)
    # 计算单通道灰度级累计值
    rs, gs, bs = 0, 0, 0
    for i in range(256):
        if r[i] != 0:
            rs += r[i]
            r[i] = rs
        if g[i] != 0:
            gs += g[i]
            g[i] = gs
        if b[i] != 0:
            bs += b[i]
            b[i] = bs
    # 根据灰度等级所在分位，计算新的对应值
    for x in range(h):
        for y in range(w):
            bi, gi, ri = img[x, y]
            img_h[x, y, 0] = b[bi] * 256 / (h * w) - 1
            img_h[x, y, 1] = g[gi] * 256 / (h * w) - 1
            img_h[x, y, 2] = r[ri] * 256 / (h * w) - 1
            # img_h[x, y, 0] = b[bi] * 255 / (h * w)
            # img_h[x, y, 1] = g[gi] * 255 / (h * w)
            # img_h[x, y, 2] = r[ri] * 255 / (h * w)
    # print(img_h)
    # sys.exit()
    return img_h


def cv2_histogram_equalization(img):
    b, g, r = cv2.split(img)
    bh = cv2.equalizeHist(b)
    gh = cv2.equalizeHist(g)
    rh = cv2.equalizeHist(r)
    return cv2.merge((bh, gh, rh))


# 卷积
def convolution(img, kernel, stride, padding, bias):
    h, w = img.shape[:2]
    fh, fw = kernel.shape[:2]
    # 将数据进行填充
    img_padding = np.zeros((h + 2*padding, w + 2*padding, 3), img.dtype)
    for i in range(h):
        for j in range(w):
            img_padding[i+padding, j+padding] = img[i, j]
#     开始卷积
    hp, wp = img_padding.shape[:2]
    img_res = np.zeros((int((hp - fh)/stride + 1), int((wp - fw)/stride + 1), 1), img.dtype)
    hr, wr = img_res.shape[:2]
    for x in range(hr):
        for y in range(wr):
            rs = 0
            # 所有对应点进行乘积并相加
            for i in range(fh):
                for j in range(fw):
                    fr, fg, fb = kernel[i, j]
                    pr, pg, pb = img_padding[x*stride + i, y*stride + j]
                    rs += int(fr) * int(pr) + int(fg) * int(pg) + int(fb) * int(pb)
    #         # 如果有偏移量，则加上偏移量
            img_res[x, y] = rs + bias
    # print(img_res.shape, img_res)
    # sys.exit()
    return img_res


def pca_self(x ,keepnum):
    #     1.中心化
    h, w = x.shape
    # 求每列平均数
    x0_sum, x1_sum, x2_sum, x3_sum =0, 0, 0, 0
    for i in range(h):
        x0, x1, x2, x3 = x[i]
        x0_sum += x0
        x1_sum += x1
        x2_sum += x2
        x3_sum += x3
    x0_avg = x0_sum / h
    x1_avg = x1_sum / h
    x2_avg = x2_sum / h
    x3_avg = x3_sum / h
    # 开始中心化
    for i in range(h):
        x0, x1, x2, x3 = x[i]
        x[i] = x0 - x0_avg,  x1 - x1_avg,  x2 - x2_avg,  x3 - x3_avg
    # 求协方差矩阵
    x_cov = 1/(h-1)*(np.dot(x.T, x))
    # 求特征向量,特征值（投影方差）
    values, vectors = np.linalg.eig(x_cov)
    # 根据指定特征数，选取特征向量
    pick_v = np.zeros((w, keepnum))
    sort_v = np.argsort(values)
    pick_part = 0
    all_part = sum(values)
    for i in range(w-keepnum, len(values)):
        # pick_part += values[sort_v[i]]
        # print(values[sort_v[i]]/all_part)
        for j in range(w):
            pick_v[j, w-i-1] = vectors[sort_v[i]][j]
    res = np.dot(x, pick_v)
    # 总贡献率
    # ratio = pick_part/all_part*100
    # print(pick_v)
    return res


if __name__ == '__main__':
    img = cv2.imread('lenna.jpg')
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dst = cv2.equalizeHist(img_gray)
    # 直方图
    # hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # cv2直方图均衡化
    # dst = cv2_histogram_equalization(img)
    # 个人实现直方图均衡化
    # dst = histogram_equalization(img)
    # hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # plt.figure()
    # plt.hist(dst.ravel(), 256)
    # plt.show()
    # 卷积
    # kernel = np.zeros((3, 3, 3), img.dtype)
    # for i in range(3):
    #     for j in range(3):
    #         kernel[i, j, 0] = -1
    #         kernel[i, j, 1] = -1
    #         kernel[i, j, 2] = -1
    # kernel[1, 1, 0] = 9
    # kernel[1, 1, 1] = 9
    # kernel[1, 1, 2] = 9
    #
    # dst = convolution(img, kernel, 1, 1, 0)
    # cv2.imshow('img', img)
    # cv2.imshow('dst', dst)
    # cv2.waitKey()
    x, y = load_iris(return_X_y=True)
    # 自己实现
    res = pca_self(x, 2)
    # sklearn 库
    pca = dp.PCA(n_components=2)
    res1 = pca.fit_transform(x)
    # print(pca.explained_variance_ratio_)
    # print(res,res1)
    # sys.exit()
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    hx, wx = x.shape
    for i in range(hx):
        if y[i] == 0:
            red_x.append(res[i, 0])
            red_y.append(res[i, 1])
        if y[i] == 1:
            blue_x.append(res[i, 0])
            blue_y.append(res[i, 1])
        else:
            green_x.append(res[i, 0])
            green_y.append(res[i, 1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()
