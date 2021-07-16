import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.subplot(224)
plt.hist(dst.ravel(), 256)
plt.title('grayscale histogram equalization')

plt.subplot(223)
plt.hist(gray.ravel(), 256)
plt.title('grayscale histogram')

plt.subplot(221)
plt.imshow(gray, cmap='gray')
plt.title('gray')

plt.subplot(222)
plt.imshow(dst, cmap='gray')
plt.title('gray equalization')

plt.show()

cv2.imshow('Histogram Equalization', np.hstack([gray, dst]))
cv2.waitKey(0)
