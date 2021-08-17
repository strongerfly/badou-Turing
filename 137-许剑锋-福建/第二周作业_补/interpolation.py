import cv2
import numpy as np
from math import floor
import matplotlib.pyplot as plt


def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nearest_interpolation(img, new_width, new_height):
    width, height, channel = img.shape
    empty_img = np.zeros((new_width, new_height, 3), np.uint8)
    ratio_w = new_width / width
    ratio_h = new_height / height
    for channel in range(channel):
        for i in range(new_width):
            for j in range(new_height):
                x = int(i/ratio_w)
                y = int(j/ratio_h)
                empty_img[i, j, channel] = img[x, y, channel]
    return empty_img


def binary_interpolation(img, dist_width, dist_height):
    src_height, src_width, channel = img.shape
    empty_img = np.zeros((dist_height, dist_width, channel), np.uint8)
    ratio_w = float(dist_width) / src_width
    ratio_h = float(dist_height) / src_height
    for chan in range(channel):
        for h in range(dist_height-1):
            for w in range(dist_width-1):
                # 平移，使得几何中心位置一致
                src_x = (w + 0.5) / ratio_w - 0.5
                src_y = (h + 0.5) / ratio_h - 0.5
                # 取整，四个角坐标值
                src_x0 = int(floor(src_x))
                # src_x1 = int(np.ceil(src_x))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y0 = int(floor(src_y))
                # src_y1 = int(np.ceil(src_y))
                src_y1 = min(src_y0 +1 , src_height - 1)
                # 计算中间点的像素值
                # if src_x0 != src_x1 and src_y1 != src_y0:
                # temp0 = ((src_x1 - src_x) * img[src_y0, src_x0, chan] + (src_x - src_x0) * img[src_y0, src_x1, chan])
                temp0 = (img[src_y0, src_x0, chan] * (src_x1 - src_x) + img[src_y0, src_x1, chan] * (src_x - src_x0))/(src_x1 - src_x0)

                # temp1 = ((src_x1 - src_x) * img[src_y1, src_x0, chan] + (src_x - src_x0) * img[src_y1, src_x1, chan])
                temp1 = (img[src_y1, src_x0, chan] * (src_x1 - src_x) + img[src_y1, src_x1, chan] * (src_x - src_x0))/(src_x1 - src_x0)
                # 计算插值
                dist_pix = int(((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1))/(src_y1 - src_y0)

                # temp0 = ((src_x1 - src_x) * img[src_y0, src_x0,
                #                                 i] + (src_x - src_x0) * img[src_y0, src_x1, i]) / (src_x1 - src_x0)
                # temp1 = (src_x1 - src_x) * img[src_y1, src_x0,
                #                                i] + (src_x - src_x0) * img[src_y1, src_x1, i] / (src_x1 - src_x0)
                # dst_img[dst_y, dst_x, i] = int(
                #     (src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1) / (src_y1 - src_y0)
                empty_img[h, w, chan] = dist_pix
    return empty_img


def rgb2gray(img):
    r, g, b = img[:, :, 0], img[:, :, 1],img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape((gray.shape[0], gray.shape[1], 1))


img = cv2.imread('../img/lenna.png')
# large_img = binary_interpolation(img, 1000, 1000)
# large_img = nearest_interpolation(img, 1000, 1000)
# print(large_img.shape)
# show(large_img, 'large')

large2_img = cv2.resize(img, (1000, 1000))
show(large2_img, 'lareg2')
print(large2_img.shape)
# print(large_img == large2_img)
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap( 'gray' ))
plt.show()
print(gray.shape)
# show(gray, "gray")
