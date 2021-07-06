import cv2
import numpy as np


def bilinear(img, dstsize):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = dstsize[1], dstsize[0]
    sh = src_h / dst_h
    sw = src_w / dst_w
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,channels),dtype=np.uint8)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * sw - 0.5
                src_y = (dst_y + 0.5) * sh - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp1 = (src_x1-src_x)*img[src_y0,src_x0,i] + (src_x-src_x0)*img[src_y0,src_x1,i]
                temp2 = (src_x1-src_x)*img[src_y1,src_x0,i] + (src_x-src_x0)*img[src_y1,src_x1,i]

                dst_img[dst_y,dst_x,i] = (src_y1-src_y)*temp1 + (src_y-src_y0)*temp2
    return dst_img

img = cv2.imread("lenna.png")
dst_img = bilinear(img,(800,800))
cv2.imshow("dst_img",dst_img)
cv2.imshow("img",img)
cv2.waitKey()