# -*- coding:utf-8 -*-
"""
cv2.resize  interpolation: 插值方式，可选值如下
INTER_NEAREST 最近邻插值
INTER_LINEAR 双线性插值（默认设置）
INTER_AREA 使用像素区域关系进行重采样。
INTER_CUBIC 4x4像素邻域的双三次插值
INTER_LANCZOS4 8x8像素邻域的Lanczos插值
"""
import cv2


def average_hash(img):
    """均值 hash """
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sum_pixel, hash_list = 0, []
    for i in range(8):
        for j in range(8):
            sum_pixel += gray_img[i][j]
    avg_pixel = sum_pixel / 64
    for i in range(8):
        for j in range(8):
            hash_list.append('1' if gray_img[i][j] > avg_pixel else '0')
    return "".join(hash_list)


def difference_hash(img):
    """ 差值 hash """
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_list = []
    for i in range(8):
        for j in range(8):
            hash_list.append('1' if gray_img[i][j] > gray_img[i][j + 1] else '0')
    return "".join(hash_list)


def compare_hash(h1, h2):
    if len(h1) != len(h2):
        raise ValueError('hash值长度不一样，无法进行比较')
    mc = 0
    for i in range(len(h1)):
        if h1[i] != h2[i]:
            mc += 1
    return mc


if __name__ == '__main__':
    src1 = cv2.imread("lenna.png")
    src2 = cv2.imread("lenna_noise.png")
    print("{} 均值hash {}".format("*" * 20, "*" * 20))
    hash1 = average_hash(src1)
    hash2 = average_hash(src2)
    print("hash1:", hash1)
    print("hash2:", hash2)
    print("汉明距离:", compare_hash(hash1, hash2))

    print("{} 差值hash {}".format("*" * 20, "*" * 20))
    hash3 = difference_hash(src1)
    hash4 = difference_hash(src2)
    print("hash3:", hash3)
    print("hash4:", hash4)
    print("汉明距离:", compare_hash(hash3, hash4))