import cv2
import numpy as np

img = cv2.imread('../img/lenna.png', 0)


def mean_hash(img):
    reshape_img = cv2.resize(img, (8, 8))
    mean = np.mean(reshape_img)
    hash_array = []
    for h in range(reshape_img.shape[0]):
        for w in range(reshape_img.shape[1]):
            if reshape_img[h, w] > mean:
                hash_array.append(1)
            else:
                hash_array.append(0)
    return hash_array


def residual_hash(image):
    reshape_img = cv2.resize(image, (8, 9))
    residual_array = []
    for h in range(reshape_img.shape[0]):
        for w in range(reshape_img.shape[1] - 1):
            if reshape_img[h, w+1] > reshape_img[h, w]:
                residual_array.append(1)
            else:
                residual_array.append(0)
    return residual_array


d_array = mean_hash(img)
r_array = residual_hash(img)
print(d_array)
print(r_array)