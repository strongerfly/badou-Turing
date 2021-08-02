from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

# 加载图像
plt.subplot(221)
image = plt.imread('lenna.png')
plt.imshow(image)
print("---- image lenna ----")
print(image)

# 灰度化
image_gray = rgb2gray(image)
plt.subplot(222)
plt.imshow(image_gray, cmap='gray')
print("---- gray image ----")
print(image_gray)

# 二值化
image_binary = np.where(image_gray <=0.5 , 0, 1)
print("---- binary image ----")
print(image_binary)
plt.subplot(223)
plt.imshow(image_binary, cmap='gray')

plt.show()