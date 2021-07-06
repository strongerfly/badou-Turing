
import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2grey(imagPath):
    img = cv2.imread(imagPath)
    h,w = img.shape[:2]
    grey = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            bgr = img[i, j]
            grey[i, j] = int(bgr[0] * 0.11 + bgr[1] * 0.59 + bgr[2] * 0.3)
    return grey

def rgb2binary(imagePath):
    grey = rgb2grey(imagePath)
    h, w = grey.shape[:2]
    binary = np.zeros([h, w], grey.dtype)
    t = np.mean(grey)
    for i in range(h):
        for j in range(w):
            if grey[i,j] <= t:
                binary[i, j] = 0
            else:
                binary[i, j] = 1
    return binary

imagPath = "lenna.png"
img_rgb = plt.imread(imagPath)
img_grey = rgb2grey(imagPath)
img_binary = rgb2binary(imagPath)
plt.subplot(131)
plt.imshow(img_rgb, cmap='gray')
plt.subplot(132)
plt.imshow(img_grey, cmap='gray')
plt.subplot(133)
plt.imshow(img_binary, cmap='gray')
plt.savefig("rgb_gray_binary")
plt.show()
