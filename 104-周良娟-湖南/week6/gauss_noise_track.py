#!/usr/bin/env python
# encoding=gbk

'''
Canny��Ե��⣺�Ż��ĳ���
'''
import cv2
import numpy as np

def gauss_noise(sigma, mu=0):
    h, w = gray.shape
    noise = sigma * np.random.randn(h, w) + mu
    new_gray = gray + noise
    new_gray[new_gray > 255] = 255
    new_gray[new_gray < 0] = 0
    cv2.imshow('gauss noise demo',new_gray.astype(np.uint8))

sigma = 5
mu = 0
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #ת����ɫͼ��Ϊ�Ҷ�ͼ
cv2.namedWindow('gauss noise demo')

#���õ��ڸ�,
'''
�����ǵڶ���������cv2.createTrackbar()
����5����������ʵ������������������ʹ����֪����ʲô��˼��
��һ�������������trackbar���������
�ڶ��������������trackbar����������������
�����������������trackbar��Ĭ��ֵ,Ҳ�ǵ��ڵĶ���
���ĸ������������trackbar�ϵ��ڵķ�Χ(0~count)
������������ǵ���trackbarʱ����(�Ļص�������
'''
# cv2.createTrackbar('sigma','gauss noise demo',sigma, 100, gauss_noise)
cv2.createTrackbar('mu','gauss noise demo',mu, 100, gauss_noise)
gauss_noise(0)  # initialization
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()
