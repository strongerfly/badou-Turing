#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ����������ͺ���ֵ1��
�������������ͺ���ֵ2��
'''

img = cv2.imread("lenna.png", 1)
#img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
#cv2.imshow("Origin", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("canny", cv2.Canny(gray, 100, 200))
cv2.waitKey()
cv2.destroyAllWindows()
