import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(gray.ravel(), 256)
# hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.plot(hist)
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.xlim([0, 270])
plt.ylim([0, 4000])
plt.show()