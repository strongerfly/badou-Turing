import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_function(img_input):
    h, w = img_input.shape[:2]
    hist = cv2.calcHist([img_input], [0], None, [256], [0, 256])
    dst_img = np.zeros((h, w), np.uint8)
    init = 0
    for i in range(256):
        init += hist[i]
        y_index, x_index = np.where(img_input == i)
        v = np.round(init/(h*w)*256-1)
        if v < 0:
            v = 0
        dst_img[y_index, x_index] = v
    return dst_img


img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_histogram = histogram_function(gray)       # 自己写的函数
#img_histogram = cv2.equalizeHist(gray)        # API调用
plt.figure()
plt.hist(img_histogram.ravel(), 256)
plt.show()
cv2.imshow("1", img_histogram)
cv2.waitKey()


# 三通道直方图
img = cv2.imread("lenna.png", 1)
b, g, r = cv2.split(img)
dst_b = cv2.equalizeHist(b)
dst_g = cv2.equalizeHist(g)
dst_r = cv2.equalizeHist(r)
result = cv2.merge((dst_b, dst_g, dst_r))
cv2.imshow("1", result)
cv2.imshow("2", img)
cv2.waitKey()

