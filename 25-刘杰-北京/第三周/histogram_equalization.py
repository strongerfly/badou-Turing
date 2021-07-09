#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：histogram_equalization.py
@Author  ：luigi
@Date    ：2021/7/6 上午10:33 
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def equalize(gray):
    """ 直方图均衡化实现方法

    :param gray: 灰度图像
    :type gray: np.ndarray(np.uint8)
    :return: 直方图均衡化的目标图像
    :rtype: np.ndarray(np.uint8)
    """
    h,w = gray.shape
    unique_pixel_origin, counts = np.unique(gray, return_counts=True)

    translation = np.cumsum(counts*256/(h*w)) - 1
    #translation里包含小数，还有可能出现负数，需要做以下处理
    unique_pixel_target = np.round(np.where(translation<0,0,translation)).astype(np.uint8)

    # 参考：https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
    target = np.arange(gray.max()+1)
    target[unique_pixel_origin] = unique_pixel_target
    target = target[gray]

    return target

def main():

    # 测试老师提供的例子
    gray = np.array((1,3,9,9,8,2,1,3,7,3,3,6,0,6,4,6,8,2,0,5,2,9,2,6,0)).reshape((5,5))
    result = equalize(gray)
    print(result)

    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--path',required=True,help="image path for input")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    target_1d = equalize(gray)

    origin = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    b,g,r = cv2.split(image)
    _b = equalize(b)
    _g = equalize(g)
    _r = equalize(r)
    target_3d = cv2.merge([_r,_g,_b])

    plt.subplot(221)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("the original gray picture")
    plt.imshow(gray,cmap='gray')
    plt.subplot(222)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("the equalized gray picture")
    plt.imshow(target_1d,cmap='gray')
    plt.subplot(223)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("the original color picture")
    plt.imshow(origin)
    plt.subplot(224)
    plt.tight_layout(pad=2.0)  # subplot间距
    plt.title("the equalized color picture")
    plt.imshow(target_3d)
    plt.show()

if __name__ == '__main__':
    main()


