#!/usr/bin/env python
# encoding=gbk

import  cv2
from matplotlib import pyplot as plt

# '''

# equalizeHist��ֱ��ͼ���⻯
# ����ԭ�ͣ� equalizeHist(src, dst=None)
# src��ͼ�����(��ͨ��ͼ��)
# dst��Ĭ�ϼ���
# '''

def cv_show(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv ����
img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_hist = cv2.equalizeHist(gray)
cv_show("image",gray_hist)

#��ɫͼ���⻯
b,g,r = cv2.split(img)
img_b = cv2.equalizeHist(b)
img_g = cv2.equalizeHist(g)
img_r = cv2.equalizeHist(r)

img2 = cv2.merge((img_b,img_g,img_r))
cv_show("image",img_b)

# ֱ��ͼ
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
# imag
#
#     imaes:�����ͼ��
#     channels:ѡ��ͼ���ͨ��
#     mask:��Ĥ����һ����С��imageһ����np���飬���а���Ҫ����Ĳ���ָ��Ϊ1������Ҫ����Ĳ���ָ��Ϊ0��һ������ΪNone����ʾ��������ͼ��
#     histSize:ʹ�ö��ٸ�bin(����)��һ��Ϊ256
#     ranges:����ֵ�ķ�Χ��һ��Ϊ[0,255]��ʾ0~255
#
# �������������������ùܡ�
# ע�⣬����mask�������ĸ�������Ҫ��[]�š�

#Test
hist = cv2.calcHist([gray_hist], [0], None, [256], [0, 256])
#hist = cv2.calcHist([img_b,img_g],[0,0],None,[256,256],[0,255,0,255])

# plt.plot(hist)
plt.figure()
# plt.hist(gray_hist.ravel(),bins=256)
plt.show()
