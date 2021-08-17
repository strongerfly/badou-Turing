import cv2
import numpy as np

def nearest(img,dstsize):
    h,w,channels = img.shape
    dst_img = np.zeros((dstsize[1],dstsize[0],channels),dtype=np.uint8)
    sh = h/dstsize[1]
    sw = w/dstsize[0]
    for i in range(dstsize[1]):
        for j in range(dstsize[0]):
            x = int(j*sw)
            y = int(i*sh)
            dst_img[i,j] = img[y,x]
    return dst_img

img = cv2.imread('lenna.png')
dst_img = nearest(img,(800,800))
cv2.imshow("dst_img",dst_img)
cv2.waitKey()