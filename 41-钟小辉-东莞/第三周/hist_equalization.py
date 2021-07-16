#!/usr/bin/env python
# encoding=gbk

import  cv2
from matplotlib import pyplot as plt

# '''

# equalizeHist—直方图均衡化
# 函数原型： equalizeHist(src, dst=None)
# src：图像矩阵(单通道图像)
# dst：默认即可
# '''

def cv_show(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv 方法
img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_hist = cv2.equalizeHist(gray)
cv_show("image",gray_hist)

#彩色图均衡化
b,g,r = cv2.split(img)
img_b = cv2.equalizeHist(b)
img_g = cv2.equalizeHist(g)
img_r = cv2.equalizeHist(r)

img2 = cv2.merge((img_b,img_g,img_r))
cv_show("image",img_b)

# 直方图
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
# imag
#
#     imaes:输入的图像
#     channels:选择图像的通道
#     mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
#     histSize:使用多少个bin(柱子)，一般为256
#     ranges:像素值的范围，一般为[0,255]表示0~255
#
# 后面两个参数基本不用管。
# 注意，除了mask，其他四个参数都要带[]号。

#Test
hist = cv2.calcHist([gray_hist], [0], None, [256], [0, 256])
#hist = cv2.calcHist([img_b,img_g],[0,0],None,[256,256],[0,255,0,255])

# plt.plot(hist)
plt.figure()
# plt.hist(gray_hist.ravel(),bins=256)
plt.show()
