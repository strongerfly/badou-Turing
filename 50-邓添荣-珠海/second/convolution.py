import cv2
import numpy as np
from scipy import ndimage


img = cv2.imread("lenna.png", 0)
m1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
dst1 = ndimage.convolve(img, m1, cval=0.0, origin=0)                   #ndimage.convlove(intput, weights, output, mode='reflect', cval=0.0, origin=0)
                                                   #weights为卷积核或者滤波器， mode为填充方式，cval为填充的常数，origin为卷积核的中心偏移
IMG = cv2.filter2D(img, -1, m1)                    #cv2.filter2D（src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None）
                                                   #ddepth为输出图像的维度，-1为与输入相同；anchor为卷积核的中心位置，默认为（-1，-1）；delta为输出结果前对每个像素加上的偏移值；borderType与莫得相似，为填充方式
cv2.imshow("1", dst1)
cv2.imshow("2", IMG)
cv2.waitKey()

