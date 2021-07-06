"""
@author:Suhao

彩色图像的原图、灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化12345678
img = cv2.imread("lenna.png")
# 获取图片的high和wide
w, h = img.shape[:2]
# 创建一张和当前图片大小一样的单通道图片
# zeros(shape, dtype=float, order=‘C’)
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        # 取出当前high和wide中的BGR坐标，注意这里还没有转换城RGB，Gray=r*0.3+g*0.59*+b*0.11
        m = img[i, j]
        # 将BGR坐标转化为gray坐标并赋值给新图像,小数的形式
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

# 原图
plt.subplot(221)
# 读取原图的数据
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
# 展示图片于221
plt.imshow(img)
print("---image lenna----")
# 打印具体的图像数据信息0-1
print(img)

# 灰度化
# 下面rgb2gray函数相当于上面的一个配比过程
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
# 下面的np.where函数对应上面两个循环的过程
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----image_binary------")
print(img_binary)
print(img_binary.shape)

# plt.subplot(2, 2, 3)行，列，标号;从上到下，从左到右的过程
plt.subplot(224)
plt.imshow(img_binary, cmap='gray')

# 展示
plt.show()
