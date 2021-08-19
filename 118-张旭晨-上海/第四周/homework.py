# -*- coding: utf-8 -*-

import numpy
import cv2
import math
import sys
from matplotlib import pyplot as plt


def canny():
    img = cv2.imread("lenna.jpg")
    # 1、灰度化
    gray = rgb2gray(img)
    gaussian = gaussian_kernel(0.5)
    # print(gaussian)
    # 2、图像去噪
    cov_img = convolution(gray, gaussian, 2)
    # 3、边缘检测
    sobel_img, angle = sobel(cov_img)
    # 4、极大值抑制
    img_nml = nml(sobel_img, angle)
    # 5、双阈值检测
    img_check = limit_check(img_nml, 100, 200)
    # plt.figure(4)
    # plt.imshow(img_check.astype(numpy.uint8), cmap='gray')
    # plt.axis('off')
    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.imshow('cov', cov_img)
    cv2.imshow('sobel', sobel_img)
    cv2.imshow('img_nml', img_nml)
    cv2.imshow('img_check', img_check)
    cv2.waitKey()


def rgb2gray(img):
    h, w = img.shape[:2]
    img_gray = numpy.zeros((h, w), img.dtype)
    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            img_gray[i, j] = g
    return img_gray


# 高斯滤波
def gaussian_kernel(sigma):
    dim = int(numpy.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    left = 1/(2*math.pi*sigma**2)
    right = -1/(2*sigma**2)

    kernel = numpy.zeros((dim, dim), numpy.double)
    for i in range(dim):
        for j in range(dim):
            x = i - (dim + 1) / 2
            y = j - (dim + 1) / 2
            kernel[i, j] = left*math.exp(right*(x**2+y**2))
    return kernel/kernel.sum()


def sobel(img):
    sobel_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = numpy.array([[1, 2, 1], [0, 0, 0], [1, 2, 1]])
    img_pad = numpy.pad(img, ((1, 1), (1, 1)), 'constant')
    # 图像在x方向的偏导系数
    img_x = numpy.zeros(img.shape, numpy.double)
    # 图像在Y方向的偏导系数
    img_y = numpy.zeros(img.shape, numpy.double)
    img_res = numpy.zeros(img.shape, img.dtype)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            # 图像在x方向的偏导
            img_x[i, j] = numpy.sum(img_pad[i:i + 3, j:j + 3] * sobel_x)
            # 图像在Y方向的偏导
            img_y[i, j] = numpy.sum(img_pad[i:i + 3, j:j + 3] * sobel_y)
            # 图像在x,Y方向取模（最大变化率），此时变化最大，为该点极值
            img_res[i, j] = int(math.sqrt(img_x[i, j] ** 2 + img_y[i, j] ** 2))
            # print(img_x[i, j],img_y[i, j],int(img_res[i, j]),img[i,j])
            # sys.exit()
    img_x[img_x == 0] = 0.000000001
    img_angle = img_y/img_x
    return img_res, img_angle


# 滤波
def convolution(img, kernel, padding):
    h, w = img.shape[:2]
    nh, nw = h+2*padding, w+2*padding
    kh, kw = kernel.shape[:2]
    new_img = numpy.zeros((nh, nw), img.dtype)
    for i in range(h):
        for j in range(w):
            new_img[i + padding, j + padding] = img[i, j]
    img_res = numpy.zeros((h, w), img.dtype)
    for i in range(h):
        for j in range(w):
            rs = 0
            for x in range(kh):
                for y in range(kw):
                    rs += new_img[i + x, j + y]*kernel[x, y]
            img_res[i, j] = int(rs)
    return img_res


# 非极大值抑制
def nml(img, angle):
    img_nml = numpy.zeros(img.shape, img.dtype)
    h, w = img.shape[:2]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            flag = True
            dtmp = img[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] < -1:
                n_1 = (dtmp[0, 1] - dtmp[0, 0]) / angle[i, j] + dtmp[0, 1]
                n_2 = (dtmp[2, 1] - dtmp[2, 2]) / angle[i, j] + dtmp[2, 1]
                if not (img[i, j] > n_1 and img[i, j] > n_2):
                    flag = False
            elif angle[i, j] < 0:
                n_1 = (dtmp[1, 0] - dtmp[0, 0]) * angle[i, j] + dtmp[1, 0]
                n_2 = (dtmp[1, 2] - dtmp[2, 2]) * angle[i, j] + dtmp[1, 2]
                if not (img[i, j] > n_1 and img[i, j] > n_2):
                    flag = False
            elif angle[i, j] > 1:
                n_1 = (dtmp[0, 2] - dtmp[0, 1]) / angle[i, j] + dtmp[0, 1]
                n_2 = (dtmp[2, 0] - dtmp[2, 1]) / angle[i, j] + dtmp[2, 1]
                if not (img[i, j] > n_1 and img[i, j] > n_2):
                    flag = False
            elif angle[i, j] > 0:
                n_1 = (dtmp[0, 2] - dtmp[1, 2]) * angle[i, j] + dtmp[1, 2]
                n_2 = (dtmp[2, 0] - dtmp[1, 0]) * angle[i, j] + dtmp[1, 0]
                if not (img[i, j] > n_1 and img[i, j] > n_2):
                    flag = False
            if flag:
                img_nml[i, j] = img[i, j]
    return img_nml


def limit_check(img, min, max):
    h, w = img.shape[:2]
    img_n = numpy.copy(img)
    edge_points = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i, j] > max:
                img_n[i, j] = 255
                edge_points.append([i, j])
            elif img[i, j] < min:
                img_n[i, j] = 0
    while not len(edge_points) == 0:
        temp_1, temp_2 = edge_points.pop()  # 出栈
        a = img_n[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < max) and (a[0, 0] > min):
            img_n[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            edge_points.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < max) and (a[0, 1] > min):
            img_n[temp_1 - 1, temp_2] = 255
            edge_points.append([temp_1 - 1, temp_2])
        if (a[0, 2] < max) and (a[0, 2] > min):
            img_n[temp_1 - 1, temp_2 + 1] = 255
            edge_points.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < max) and (a[1, 0] > min):
            img_n[temp_1, temp_2 - 1] = 255
            edge_points.append([temp_1, temp_2 - 1])
        if (a[1, 2] < max) and (a[1, 2] > min):
            img_n[temp_1, temp_2 + 1] = 255
            edge_points.append([temp_1, temp_2 + 1])
        if (a[2, 0] < max) and (a[2, 0] > min):
            img_n[temp_1 + 1, temp_2 - 1] = 255
            edge_points.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < max) and (a[2, 1] > min):
            img_n[temp_1 + 1, temp_2] = 255
            edge_points.append([temp_1 + 1, temp_2])
        if (a[2, 2] < max) and (a[2, 2] > min):
            img_n[temp_1 + 1, temp_2 + 1] = 255
            edge_points.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_n.shape[0]):
        for j in range(img_n.shape[1]):
            if img_n[i, j] != 0 and img_n[i, j] != 255:
                img_n[i, j] = 0
    return img_n


def cv2Canny():
    img = cv2.imread("lenna.jpg")
    img_gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 100, 200)
    cv2.imshow('canny', img_canny)
    cv2.waitKey()


def sobel_laplace():
    img = cv2.imread("lenna.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    '''
    Sobel算子
    Sobel算子函数原型如下：
    dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
    前四个是必须的参数：
    第一个参数是需要处理的图像；
    第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
    dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
    其后是可选的参数：
    dst是目标图像；
    ksize是Sobel算子的大小，必须为1、3、5、7。
    scale是缩放导数的比例常数，默认情况下没有伸缩系数；
    delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
    borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
    '''
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 对x求导
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # 对y求导

    # Laplace 算子
    img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

    # Canny 算子
    img_canny = cv2.Canny(img_gray, 100, 150)

    plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
    plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
    plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
    plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")
    plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
    plt.show()



if __name__ == '__main__':
    # canny()
    # cv2Canny()
    sobel_laplace()