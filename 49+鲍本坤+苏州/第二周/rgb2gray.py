

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import turtle

# 灰度化

img = cv2.imread("lenna.png")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("image show gray", gray)
plt.imshow(gray)
plt.show()

h,w=img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]= int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print(img_gray)
print("img show gray:%s" %img_gray)
cv2.imshow("show gray",img_gray)
cv2.waitKey()  #报纸显示图像的界面，直到任意按键触发
# plt.imshow(img_gray)
# plt.show()


