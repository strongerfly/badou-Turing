#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：convolution_manual.py
@Author  ：luigi
@Date    ：2021/7/6 下午4:40 
'''

import numpy as np

def convolute(gray, kernel,stride=1,padding=0,mode='valid'):

    h, w = gray.shape
    candicate = gray
    if padding>0:
        if mode == 'full':
            candicate = np.zeros((h+2*padding,w+2*padding))
            candicate[padding:h + padding, padding:w + padding] = gray
            h += 2*padding
            w += 2*padding

        if mode == 'same':
            kernel_radius = kernel.shape[0]//2
            # 增加的边缘要比kernel半径大，才有可能得到same的结果
            assert stride * kernel_radius<=padding
            candicate = np.zeros((h+2*stride,w+2*stride))
            candicate[stride:h + stride, stride:w + stride] = gray
            h += 2*stride
            w += 2*stride

    m,n = kernel.shape
    h_new,w_new = int((h-m)/stride+1),int((w-n)/stride+1)
    assert h_new>=0 and w_new>=0

    print("the candicated area is:\n{}".format(candicate))
    target = np.zeros((h_new,w_new))
    for i in range(h_new):
        for j in range(w_new):
            target[i,j] = np.sum(candicate[i:i+m,j:j+n]*kernel)
    return target

def main():
    gray = np.arange(49).reshape((7,7))
    kernel = np.arange(25).reshape((5,5))
    target = convolute(gray,kernel,stride=1,padding=4,mode='same')
    print("the output is: \n{}".format(target))

if __name__ == '__main__':
    main()





