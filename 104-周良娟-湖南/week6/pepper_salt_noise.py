#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np


def peper(snp):
    '''
    :param snp: 信噪比
    :return:
    '''
    snp  = snp * ratio
    h, w = gray.shape
    arr = np.random.choice([0, 255], int(round(h * w * snp)))    # [0, 255, 255, 0, 0, 255] 随机出现
    # 将这些点随机加入到图像中
    # 随机选择i,j
    x = np.random.choice(h, int(round(h * w * snp)))
    y = np.random.choice(w, int(round(h * w * snp)))
    for i, j, k in zip(x, y, arr):
        gray[i,j] = k

    cv2.imshow('pepper salt noise demo', gray.astype(np.uint8))


snp = 0
ratio = 0.01
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
cv2.namedWindow('pepper salt noise')

# # 设置调节杠,
# '''
# 下面是第二个函数，cv2.createTrackbar()
# 共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
# 第一个参数，是这个trackbar对象的名字
# 第二个参数，是这个trackbar对象所在面板的名字
# 第三个参数，是这个trackbar的默认值,也是调节的对象
# 第四个参数，是这个trackbar上调节的范围(0~count)
# 第五个参数，是调节trackbar时调用(的回调函数名
# '''
cv2.createTrackbar('sigmal noise raito', 'pepper salt noise', snp, 100, peper)
peper(0)  # initialization
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()


