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
for i in range(new_h):
    for j in range(new_w):
        x = int(i / scale) if i / scale - i // scale < mid else int(i / scale) + 1
        y = int(j / scale) if j / scale - j // scale < mid else int(j / scale) + 1
        img_new[i, j] = img[x, y]
print("new:")
print(img_new)
cv2.imwrite("lenna_nearest_interp.png", img_new)
