import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image


img = cv2.imread("lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.title("color")
plt.imshow(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.title("gray")
plt.imshow(img_gray, cmap='gray')

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.title("binary")
plt.imshow(img_binary, cmap='gray')

plt.show()