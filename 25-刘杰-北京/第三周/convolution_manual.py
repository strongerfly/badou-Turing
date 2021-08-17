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
            # 输入大小与输出大小一样，应该保证stride = 1
            assert stride == 1
            least_padding = (kernel.shape[0] - 1)//2
            # 输入中的padding参数，应该大于等于same模式最少需要的padding数
            assert least_padding <= padding
            candicate = np.zeros((h+2*least_padding,w+2*least_padding))
            candicate[least_padding:h + least_padding, least_padding:w + least_padding] = gray
            h += 2*least_padding
            w += 2*least_padding

    m,n = kernel.shape
    h_new,w_new = int((h-m)/stride+1),int((w-n)/stride+1)
    assert h_new>=0 and w_new>=0

    # print("the candicated area is:\n{}".format(candicate))
    target = np.zeros((h_new,w_new))
    for i in range(h_new):
        for j in range(w_new):
            target[i,j] = np.sum(candicate[i:i+m,j:j+n]*kernel)
    return target

def main():
    gray = np.arange(49).reshape((7,7))
    kernel = np.arange(25).reshape((5,5))
    target = convolute(gray,kernel,stride=1,padding=2,mode='same')
    # print("the output is: \n{}".format(target))

if __name__ == '__main__':
    main()





