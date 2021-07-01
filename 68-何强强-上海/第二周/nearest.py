# -*- coding:utf-8 -*-
"""
最邻近插值实现
"""
import cv2
import numpy as np


def enlarge_img(img, new_width, new_height):
    height, width, channel = img.shape
    # 默认dtype为 float64, 当使用默认类型时，图片为空白
    new_img = np.zeros((new_height, new_width, channel), np.uint8)

    scale_height = new_height / height
    scale_width = new_width / width

    for h in range(new_height):
        for w in range(new_width):
            oh = round(h / scale_height)
            ow = round(w / scale_width)
            new_img[h][w] = img[oh][ow]
    return new_img


def test(file_name):
    src = cv2.imread(file_name)
    des = enlarge_img(src, 800, 900)

    cv2.imshow("src", src)
    cv2.imshow("des", des)
    cv2.waitKey()


if __name__ == '__main__':
    test("lenna.png")
