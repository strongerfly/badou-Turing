import cv2
import numpy as np

def func(img):
    h,w = img.shape[:2]
    img_gray = np.zeros((h,w),img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            img_gray[i,j]= int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    return img_gray

img = cv2.imread('lenna.png')
dst_img = func(img)
cv2.imshow("rgb2gray",dst_img)
cv2.waitKey()

