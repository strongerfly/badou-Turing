# -*- coding:utf-8 -*-
"""
透视变换是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
我们常说的仿射变换是透视变换的一个特例。
透视变换的目的就是把现实中为直线的物体，在图片上可能呈现为斜线，通过透视变换转换成直线 的变换。
仿射变换（Affine Transformation或Affine Map），又称为仿射映射，是指在几何中，
图像进行从 一个向量空间进行一次线性变换和一次平移，变换为到另一个向量空间的过程。
"""
import cv2
import numpy as np


def perspective_transformation_by_cv2():
    """ open-cv 提供的透视变换方法 """
    img = cv2.imread("qx1.jpg", flags=1)
    cv2.imshow("src img", img)
    src = np.float32([[315, 39], [416, 50], [236, 476], [407, 477]])
    dst = np.float32([[50, 0], [190, 0], [50, 488], [190, 488]])
    # 获取透视变换矩阵
    warp_matrix = cv2.getPerspectiveTransform(src, dst)
    print("warp_matrix: {}".format(warp_matrix))
    # 使用矩阵warp_matrix进行变换
    dst_img = cv2.warpPerspective(img, warp_matrix, (300, 488))
    cv2.imshow("dst img", dst_img)
    cv2.waitKey()


def calc_warp_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    point_nums = src.shape[0]

    # A * warpMatrix = B
    arr_a = np.zeros((2 * point_nums, 8))
    arr_b = np.zeros((2 * point_nums, 1))
    for i in range(0, point_nums):
        a_i, b_i = src[i], dst[i]
        arr_a[2 * i] = [a_i[0], a_i[1], 1, 0, 0, 0, -a_i[0] * b_i[0], -a_i[1] * b_i[0]]
        arr_a[2 * i + 1] = [0, 0, 0, a_i[0], a_i[1], 1, -a_i[0] * b_i[1], -a_i[1] * b_i[1]]
        arr_b[2 * i] = b_i[0]
        arr_b[2 * i + 1] = b_i[1]
    # A^ 为 A的逆矩阵
    # A^ * A * warpMatrix = A^ * B  => warpMatrix = A^ * B
    # arr_a_inv = np.mat(arr_a).I
    # warp_matrix = arr_a_inv * arr_b
    arr_a_inv = np.linalg.inv(arr_a)
    # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warp_matrix = np.dot(arr_a_inv, arr_b)
    # 将a33塞回原来的位置
    warp_matrix = warp_matrix.ravel()
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    return warp_matrix.reshape(3, 3)


if __name__ == '__main__':
    # perspective_transformation_by_cv2()
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    wm = calc_warp_matrix(src, dst)
    print(wm)