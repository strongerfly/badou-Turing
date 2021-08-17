# -*- coding: utf-8 -*-
"""
双线性插值
"""
import cv2
import numpy as np


def change(img, new_height, new_width):
    height, width, channel = img.shape
    new_img = np.zeros((new_height, new_width, 3), np.uint8)
    sh = height / new_height
    sw = width / new_width
    for i in range(3):
        for h in range(new_height):
            for w in range(new_width):
                # 找到对应在原图像上的坐标
                src_x = (w + 0.5) * sw - 0.5
                src_y = (h + 0.5) * sh - 0.5

                # 左边取较大值，右边取较小值，防止越界
                src_x1 = max(int(np.floor(src_x)), 0)
                src_x2 = min(src_x1 + 1, width - 1)
                src_y1 = max(int(np.floor(src_y)), 0)
                src_y2 = min((src_y1 + 1, height - 1))
                try:
                    t0 = (src_y2 - src_y) * ((src_x2 - src_x) * img[src_y1, src_x1, i] + (src_x - src_x1) * img[src_y1, src_x2, i])
                    t1 = (src_y - src_y1) * ((src_x2 - src_x) * img[src_y2, src_x1, i] + (src_x - src_x1) * img[src_y2, src_x2, i])
                    new_img[h, w, i] = t0 + t1
                except IndexError as e:
                    print(src_y1, src_y2, src_x1, src_x2)
    return new_img


if __name__ == '__main__':
    src = cv2.imread("lenna.png")
    cv2.imshow("src", src)

    des = change(src, 700, 700)
    cv2.imshow("des", des)

    cv2.waitKey()
