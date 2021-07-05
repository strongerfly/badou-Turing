from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = rgb2gray(img_gray)
img_1or0 = np.where(img_gray >= 0.5, 1, 0)
img_1or0 = np.array(img_1or0, np.uint8)
img_1or0 = 255*img_1or0
# C
cv2.imshow("1", img)
cv2.imshow("2", img_gray)
cv2.imshow("3", img_1or0)
# 3图显示

img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
img_1or0 = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_1or0, cmap='gray')

plt.show()