#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：perspectiveTransform.py
@Author  ：luigi
@Date    ：2021/7/20 上午10:26 
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def perspectiveMatrix(src,dst):
    """ 获取warp matrix

    :param src: 原始图像的4点
    :type src: np.ndarray
    :param dst: 目标图像的4点
    :type dst: np.ndarray
    :return: warp matrix
    :rtype: numpy.ndarray
    """
    assert src.shape == dst.shape and src.shape[0] >= 4

    xs_0, xs_1, xs_2, xs_3 = src[:,0]
    ys_0, ys_1, ys_2, ys_3 = src[:,1]

    xd_0, xd_1, xd_2, xd_3 = dst[:,0]
    yd_0, yd_1, yd_2, yd_3 = dst[:,1]

    A = np.matrix([
        [xs_0, ys_0, 1, 0, 0, 0, -xs_0 * xd_0, -ys_0 * xd_0],
        [0, 0, 0, xs_0, ys_0, 1, -xs_0 * yd_0, -ys_0 * yd_0],
        [xs_1, ys_1, 1, 0, 0, 0, -xs_1 * xd_1, -ys_1 * xd_1],
        [0, 0, 0, xs_1, ys_1, 1, -xs_1 * yd_1, -ys_1 * yd_1],
        [xs_2, ys_2, 1, 0, 0, 0, -xs_2 * xd_2, -ys_2 * xd_2],
        [0, 0, 0, xs_2, ys_2, 1, -xs_2 * yd_2, -ys_2 * yd_2],
        [xs_3, ys_3, 1, 0, 0, 0, -xs_3 * xd_3, -ys_3 * xd_3],
        [0, 0, 0, xs_3, ys_3, 1, -xs_3 * yd_3, -ys_3 * yd_3]
        ],dtype=np.float32)

    b = dst.reshape((8,1))
    X = np.linalg.solve(A,b)
    matrix = np.append(X,1).reshape((3,3))

    return matrix

def perspectiveTransform(gray, matrix, size):
    """ 通过warp matrix的逆矩阵，得到目标图像在原始图像中的坐标，进行透视变换

    :param gray: 原始图像
    :type gray: np.ndarray
    :param matrix: warp mastrix
    :type matrix: np.ndarray
    :param size: 目标图像大小
    :type size: tuple
    :return: 透视变换后的图像
    :rtype: np.ndarray
    """
    #目标图像中的坐标范围（！非矩阵坐标）
    coordX, coordY = np.indices(size)
    # 这里求warp matrix 的逆矩阵
    a11,a12,a13,a21,a22,a23,a31,a32,a33 = np.linalg.inv(matrix).reshape((9,))

    # 映射出新图像坐标在原始图像中的映射坐标
    coordinateSourceX = ((coordX*a11 + coordY*a12 + a13) / (coordX*a31 + coordY*a32 + a33)).astype(np.int16)
    coordinateSourceY = ((coordX*a21 + coordY*a22 + a23) / (coordX*a31 + coordY*a32 + a33)).astype(np.int16)

    # 这里需要切换为矩阵坐标（即y在前，x在后）
    # 还需要再想想为啥要用转置T
    target = gray[coordinateSourceY, coordinateSourceX].T
    return target

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--path",required=True, help="path for input image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = perspectiveMatrix(src, dst)

    target_manual = perspectiveTransform(gray,warpMatrix, (337, 488))

    m= cv2.getPerspectiveTransform(src, dst)
    target_cv = cv2.warpPerspective(gray, m, (337, 488))

    plt.subplot(221)
    plt.title("origin")
    plt.imshow(gray,cmap="gray")

    plt.subplot(222)
    plt.title("transform_manual")
    plt.imshow(target_manual, cmap="gray")

    plt.subplot(223)
    plt.title("transform_cv")
    plt.imshow(target_cv, cmap="gray")

    plt.show()


if __name__ == '__main__':
    main()








