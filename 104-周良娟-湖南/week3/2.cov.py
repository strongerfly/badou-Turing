# 最简单的卷积
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

import tensorflow as tf

# tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)

# def convolution1(input_, filter_, strides, padding = None):
def convolution1(input_, filter_, strides = 1, padding=None):
    """
    先实现最简单的，strides = 1
    """

    h, w = input_.shape
    f, f = filter_.shape

    out_h = int((h - f) / strides) + 1   # 输出矩阵的h，w
    out_w = int((w - f) / strides) + 1
    output_ = np.zeros([out_h, out_w])  #  输出矩阵的初始化
    for i in range(0, h - f + 1):
        for j in range(0, w - f + 1):
            output_[i,j] = np.sum(input_[i:i+f, j:j+f] * filter_)
    return output_

def convolution2(input_, filter_, strides=1, padding=None):
    """
    先实现最简单的，strides = N
    """

    h, w = input_.shape
    f, f = filter_.shape

    out_h = int((h - f) / strides) + 1  # 输出矩阵的h，w
    out_w = int((w - f) / strides) + 1
    output_ = np.zeros([out_h, out_w])  #  输出矩阵的初始化
    for i in range(0, out_h):
        for j in range(0, out_w):
            output_[i,j] = np.sum(input_[i * strides:i * strides + f,
                                  j * strides:j * strides + f] * filter_)

    return output_

def convolution3(input_, filter_, strides=1, padding = None):
    """
    先实现最简单的，strides = N
    padding = (f - 1) / 2
    """

    h, w = input_.shape
    f, f = filter_.shape
    if not padding:
        p = int((f - 1) / 2)
    else:
        p = padding

    out_h = int((h - f + 2 * p) / strides) + 1  # 输出矩阵的h，w
    out_w = int((w - f + 2 * p) / strides) + 1
    output_ = np.zeros([out_h, out_w])  #  输出矩阵的初始化
    # 对原始矩阵进行 padding
    zero_h = np.zeros((h, p))
    zero_w = np.zeros((p , w + 2 * p))
    input_ = np.column_stack([zero_h,input_, zero_h])
    input_ = np.row_stack([zero_w,input_, zero_w])
    print(input_, 'input_')
    for i in range(0, out_h):
        for j in range(0, out_w):
            output_[i,j] = np.sum(input_[i * strides:i * strides + f,
                                  j * strides:j * strides + f] * filter_)

    return output_

# 灰色图
'''
if __name__ == '__main__':
    img = cv2.imread('lenna.png', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)
    filter_ = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
    output_1 = convolution1(gray, filter_)
    output_2 = convolution2(gray, filter_, strides=3, padding=None)
    output_3 = convolution3(gray, filter_, strides=2, padding=None)
    print(output_2, 'output_2')
    print(output_3, 'output_3')

    # 画图
    cv2.imshow('cov1 lenna ', np.hstack([output_1]))
    cv2.waitKey(0)
    cv2.imshow('cov2 lenna', np.hstack([output_2]))
    cv2.waitKey(0)
    cv2.imshow('cov3 lenna', np.hstack([output_3]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
# 彩色图
'''
if __name__ == '__main__':
    img = cv2.imread('lenna.png', 1)
    (b,g,r) = cv2.split(img)
    filter_ = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
    bh = convolution3(b, filter_, strides=1, padding=None)
    gh = convolution3(g, filter_, strides=1, padding=None)
    rh = convolution3(r, filter_, strides=1, padding=None)
    result = cv2.merge((bh, gh, rh))
    cv2.imshow('color lenna cov', img)
    cv2.waitKey(0)
    cv2.imshow('color lenna cov', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

if __name__ == '__main__':
    flower = plt.imread('flower.png',1)
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    print(flower.shape)
    plt.imshow(flower,cmap = plt.cm.gray)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # data = gray.reshape(1, 512, 512, 1)
    #
    # filter_ = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]]).reshape(3, 3, 1, 1)
    # cov = tf.nn.conv2d(data, filter_, strides=1, padding="SAME")
    # print(cov.shape)



# 备注1 np插入一行或者一列
'''
# np.insert?
# np.c_[] 和 np.r_[]
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.ones(3)

c = np.c_[np.array([1,2,3]), np.array([4,5,6])]  # 按列堆叠
print(c, 'np.c_')

r = np.r_[np.array([1,2,3]),  np.array([4,5,6])]  # 直接展开  array([1, 2, 3, 4, 5, 6])
print(r, 'np.r_')

a1 = np.insert(a , 0, 1, axis=1)   # (arr, 插入位置， 元素，1列0行)
print(a1, 'a1')

a1 = np.insert(a , 0, 1, axis=0)   # (arr, 插入位置， 元素，1列0行)
print(a1, 'a1')

a2 = np.column_stack([a,b])  # 按列拼接
print(a2, 'a2 np.column_stack')

a3 = np.row_stack([a, b])    # 按行拼接
print(a3, 'a3 np.row_stack')

vstack = np.vstack([a, b])   #添加在最后一行最后一行   列数要相同才能行拼接
print(vstack, 'np.vstack')

b1 = np.ones((3,1))
hstack = np.hstack([a,b1])  #添加在最后一行最后一列  # 行数要相同才能列拼接
print(hstack, 'np.hstack')


# concatenate 维度要相同
a=np.array([[1,2],[3,4]])
b2=np.array([[5,6]])
c=np.concatenate((a,b2),axis=0)
print(c,'c')
b2 = np.zeros((2,1))
c=np.concatenate((a,b2),axis=1)
print(c, 'con axis=1')

'''