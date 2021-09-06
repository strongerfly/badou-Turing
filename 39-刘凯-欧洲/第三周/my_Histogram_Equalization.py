#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist��ֱ��ͼ���⻯
����ԭ�ͣ� equalizeHist(src, dst=None)
src��ͼ�����(��ͨ��ͼ��)
dst��Ĭ�ϼ���
'''
def my_equalizeHist(input_image):
    '''
    ����ֱ��ͼ���⻯
    :param input_image:gray scale image
    :return: gray scale image after histogram  Equalization
    '''
    H,W= input_image.shape
    dst = np.zeros((H, W), dtype=np.uint8)
    input_hist,_ = np.histogram(input_image,256,[0,256])
    #input_hist = cv2.calcHist([input_image],[0],None,[256],[0,256])
    pi_list = []        #pi = number of pixes / image size
    for i in range(0, 256):
        pi = input_hist[i] / (H * W)
        pi_list.append(pi)
        pi_equalize = sum(pi_list) * 256 - 1
        pi_equalize = max(0, pi_equalize)
        pi_equalize = min(255, pi_equalize)
        dst[input_image == i] = int(pi_equalize)
    return dst

def my_equalizeHist2(input_image):
    '''
    ����ֱ��ͼ���⻯ʹ��CDF
    :param input_image:gray scale image
    :return: gray scale image after histogram  Equalization
    '''
    H,W= input_image.shape
    input_hist,_ = np.histogram(input_image,256,[0,256])
    cdf = input_hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf[input_image]

# ��ȡ�Ҷ�ͼ��
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# �Ҷ�ͼ��ֱ��ͼ���⻯
dst = cv2.equalizeHist(gray)
my_dst = my_equalizeHist(gray)
my_dst2 = my_equalizeHist2(gray)
# ֱ��ͼ
#hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#my_hist = cv2.calcHist([my_dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(gray.ravel(), 256)
plt.hist(dst.ravel(), 256)
plt.hist(my_dst.ravel(),256)
plt.hist(my_dst2.ravel(),256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst, my_dst, my_dst2]))
cv2.waitKey(0)
