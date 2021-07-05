import cv2
import numpy as np
from skimage.color import rgb2gray


def function(img):
    h, w, c = img.shape
    near_picture = np.zeros((600, 600, c), np.uint8)
    nh = 600/h
    nw = 600/w
    for i in range(600):
        for j in range(600):
            y = np.where((i / nh) >= (int(i / nh) + 0.5), int(i / nh) + 1, int(i / nh))
            x = np.where((j / nw) >= (int(j / nw) + 0.5), int(j / nw) + 1, int(j / nw))
            near_picture[i, j] = img[y, x]
    return near_picture


img = cv2.imread("lenna.png")
img_big = function(img)
#img_gray = rgb2gray(img_big)
#cv2.imshow("1", img)
cv2.imshow("2", img_big)
#cv2.imshow("2", img_gray)
cv2.waitKey()
