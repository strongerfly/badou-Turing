# 2.双线性插值实现
import cv2
import numpy as np

def biliner_interp(img):
    src_h, src_w, src_channels = img.shape
    dist_h, dist_w = (800, 800)
    target_img = np.zeros((dist_h, dist_w, src_channels), np.uint8)
    scale_x, scale_y = float(src_w) / dist_w, float(src_h) / dist_h
    for c in range(src_channels):
        for h in range(dist_h):
            for w in range(dist_w):
                src_x = (w + 0.5) * scale_x - 0.5
                src_y = (h + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                target_img[h, w, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return target_img


img = cv2.imread('lenna.png')
biliner_zoom = biliner_interp(img)
cv2.imshow("biliner", biliner_zoom)
cv2.imshow("img", img)
cv2.waitKey(0)