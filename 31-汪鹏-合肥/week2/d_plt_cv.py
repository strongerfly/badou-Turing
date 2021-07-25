# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:11:44 2021
cv2与plt的read、show的不同
@author: wp
"""
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

def Ips(img):
    print("properties:shape:{},size:{},dtype:{}".format(img.shape,img.size,img.dtype))

filename = 'E:/ML/Badou/Homework/work0627/lenna.png'
pfig = plt.imread(filename) 
cfig = cv2.imread(filename, 1)  #cv2.IMREAD_COLOR:1 cv2.IMREAD_GRAYSCALE:0 cv2.IMREAD_UNCHANGED:-1

# cv2.namedWindow("Lenna")

# cfig = pfig[:,:,(2,1,0)]  #交换 B 和 R 通道

# b, g, r =cv2.split(cfig)
# img_rgb = cv2.merge([r, g, b])

grayImage = cv2.cvtColor(cfig, cv2.COLOR_BGR2GRAY)  # 灰度变换

# cv2.imshow("Lenna", pfig)
# cv2.waitKey(0)


# plt.imshow(cfig, cmap='gray')
# plt.show()

cv2.imshow("lenna", grayImage)
cv2.waitKey()

plt.figure()
plt.imshow(grayImage,cmap = 'gray')
plt.axis('off')
plt.show()

Ips(pfig)
Ips(cfig)
Ips(grayImage)



