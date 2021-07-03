import cv2
import numpy as np


def bilinear_interpolation(img, out_size):
    scr_h, scr_w, channel = img.shape
    dst_h, dst_w = out_size[0], out_size[1]
    if scr_h == dst_h and scr_w == dst_w:
        return img.copy()       # 得到一个一样的图片
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)
    change_x, change_y = scr_w/dst_w, scr_h/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                scr_x = (dst_x+0.5) * change_x - 0.5
                scr_y = (dst_y+0.5) * change_y - 0.5

                scr_x0 = int(np.floor(scr_x))
                scr_x1 = min(scr_x0+1, scr_w-1)
                scr_y0 = int(np.floor(scr_y))
                scr_y1 = min(scr_y0+1, scr_h-1)

                temp0 = (scr_x1-scr_x)*img[scr_y0, scr_x0, i]+(scr_x - scr_x0)*img[scr_y0, scr_x1, i]
                temp1 = (scr_x1-scr_x)*img[scr_y1, scr_x0, i]+(scr_x - scr_x0)*img[scr_y1, scr_x1, i]
                dst_img[dst_y, dst_x, i] = (scr_y1 - scr_y)*temp0 + (scr_y - scr_y0)*temp1

    return dst_img


img = cv2.imread("lenna.png")
img_bilinear = bilinear_interpolation(img, (400, 600))
cv2.imshow("1", img_bilinear)
cv2.waitKey()