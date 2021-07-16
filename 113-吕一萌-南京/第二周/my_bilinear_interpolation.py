import numpy as np
import cv2


def my_imread(path) -> np.ndarray:
    return cv2.imread(path)


img = my_imread("lenna.png")
print(img)
# print(img.shape)
ori_h, ori_w, channel = img.shape[:3]
scale = 3 / 2
new_h, new_w = int(ori_h * scale), int(ori_w * scale)
# print(img.dtype)
img_new = np.zeros((new_h, new_w, channel), img.dtype)
mid = scale / 2
for i in range(3):
    for dst_y in range(new_h):
        for dst_x in range(new_w):
            # find the origin x and y coordinates of dst image x and y
            # use geometric center symmetry
            # if use direct way, src_x = dst_x * scale_x
            src_x = (dst_x + 0.5) * scale - 0.5
            src_y = (dst_y + 0.5) * scale - 0.5

            # find the coordinates of the points which will be used to compute the interpolation
            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1, ori_w - 1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1, ori_h - 1)

            # calculate the interpolation
            temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
            temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
            img_new[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

print("new:")
print(img_new)
cv2.imwrite("lenna_bilinear_interpoaton.png", img_new)
