#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np


def peper(snp):
    '''
    :param snp: �����
    :return:
    '''
    snp  = snp * ratio
    h, w = gray.shape
    arr = np.random.choice([0, 255], int(round(h * w * snp)))    # [0, 255, 255, 0, 0, 255] �������
    # ����Щ��������뵽ͼ����
    # ���ѡ��i,j
    x = np.random.choice(h, int(round(h * w * snp)))
    y = np.random.choice(w, int(round(h * w * snp)))
    for i, j, k in zip(x, y, arr):
        gray[i,j] = k

    cv2.imshow('pepper salt noise demo', gray.astype(np.uint8))


snp = 0
ratio = 0.01
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ת����ɫͼ��Ϊ�Ҷ�ͼ
cv2.namedWindow('pepper salt noise')

# # ���õ��ڸ�,
# '''
# �����ǵڶ���������cv2.createTrackbar()
# ����5����������ʵ������������������ʹ����֪����ʲô��˼��
# ��һ�������������trackbar���������
# �ڶ��������������trackbar����������������
# �����������������trackbar��Ĭ��ֵ,Ҳ�ǵ��ڵĶ���
# ���ĸ������������trackbar�ϵ��ڵķ�Χ(0~count)
# ������������ǵ���trackbarʱ����(�Ļص�������
# '''
cv2.createTrackbar('sigmal noise raito', 'pepper salt noise', snp, 100, peper)
peper(0)  # initialization
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()


