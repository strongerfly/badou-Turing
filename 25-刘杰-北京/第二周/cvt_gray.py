#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：cvt_gray.py
@Author  ：luigi
@Date    ：2021/6/30 下午3:41 
'''

def bgr2gray(image):

    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    gray = b*0.11 + g*0.59 + r*0.3

    return gray