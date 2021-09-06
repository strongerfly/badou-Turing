from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]                # 取彩色图片的长宽
img_gray = np.zeros([h, w], img.dtype)      # 返回一个元素全为0且给定形状和类型的数组。即创建一张和当前图片大小一致的单通道图片
for i in range(h):
    for j in  range(w):
        m = img[i, j]             # 取出当前high和weight中的BGR坐标
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)     # 将BGR转换为gray坐标，赋值给img_gray
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)          # cv2.imshow("名字", 图片)
# cv2.waitKey(0)

# 原图
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
plt.title('color')
print("***image lenna***")
print(img)

# 灰度图
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.title('gray')
print("***image gray***")
print(img_gray)

# 二值化
plt.subplot(223)
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        k = img_gray[i, j]/255
        if k <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
img_binary = img_gray
plt.title('binary')
plt.imshow(img_binary, cmap='gray')


plt.show()