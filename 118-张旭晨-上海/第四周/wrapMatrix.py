# -*- coding: utf-8 -*-


import numpy
import sys


def wrapPersitiveMatrix(src, dst):
    hs, ws = src.shape[:2]
    dh, dw = dst.shape[:2]
    A = numpy.zeros((hs*ws, hs*ws))
    B = numpy.zeros((hs*ws, 1))
    for i in range(0, hs):
        A_i = src[i, :]
        B_i = dst[i, :]
        # A的2 * i列原矩阵
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,-A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        # A的 2 * i列目标值
        B[2 * i] = B_i[0]
        # A的2 * i+ 1列原矩阵
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        # A的 2 * i + 1 列目标值
        B[2 * i + 1] = B_i[1]

    # 创建矩阵
    A = numpy.mat(A)
    # print(A,B)
    # sys.exit()
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    # 之后为结果的后处理
    warpMatrix = numpy.array(warpMatrix).T[0]
    warpMatrix = numpy.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))

    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = numpy.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = numpy.array(dst)

    warpMatrix = wrapPersitiveMatrix(src, dst)
    print(warpMatrix)