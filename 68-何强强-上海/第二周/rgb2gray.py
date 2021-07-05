# -*- coding:utf-8 -*-
"""
将rgb图像转换为gray图形
"""
import cv2


def change(img):
    pass


if __name__ == '__main__':
    src = cv2.imread("lenna.png", flags=cv2.COLOR_BGR2GRAY)