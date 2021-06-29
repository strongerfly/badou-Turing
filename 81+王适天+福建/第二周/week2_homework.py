#    @author Created by Genius_Tian

#    @Date 2021/6/27

#    @Description AI学习第二周作业

import cv2
import numpy as np
import time


# 该方法是最邻近插值法
def nearest_interpolation(src_img, target_shape):
    th, tw = target_shape
    h, w, channel = src_img.shape
    empty_img = np.zeros((th, tw, channel), src_img.dtype)
    sh = th / h
    sw = tw / w
    for i in range(th):
        for j in range(tw):
            x = int(i / sh)
            y = int(j / sw)
            empty_img[i, j] = src_img[x, y]
    return empty_img


def bilinear_interpolation(src_img, target_shape):
    h, w, channel = src_img.shape
    dst_h, dst_w = target_shape[1], target_shape[0]
    if dst_h == h and dst_w == w:
        return src_img.copy()

    empty_img = np.zeros((dst_h, dst_w, channel), src_img.dtype)
    scale_x, scale_y = float(w) / dst_w, float(h) / dst_h
    for dst_y in range(dst_h):
        src_y = (dst_y + 0.5) * scale_y - 0.5
        for dst_x in range(dst_w):
            src_x = (dst_x + 0.5) * scale_x - 0.5

            src_x0 = int(np.floor(src_x))
            src_y0 = int(np.floor(src_y))

            src_x1 = min(src_x0 + 1, w - 1)
            src_y1 = min(src_y0 + 1, h - 1)
            print(img[src_y0, src_x0, :])
            r1 = (src_x1 - src_x) * img[src_y0, src_x0, :] + (src_x - src_x0) * img[src_y0, src_x1, :]
            r2 = (src_x1 - src_x) * img[src_y1, src_x0, :] + (src_x - src_x0) * img[src_y1, src_x1, :]
            empty_img[dst_y, dst_x, :] = ((src_y1 - src_y) * r1 + (src_y - src_y0) * r2).astype(int)

    return empty_img


img = cv2.imread("lenna.png")
# target_img = nearest_interpolation(img, (100, 100))
start = time.time()
interpolation = bilinear_interpolation(img, (800, 800))
end = time.time()
cv2.imshow("nearest interp", img)
print(end - start)
# cv2.imshow("image", target_img)
cv2.imshow("interpolation", interpolation)
cv2.waitKey(0)
