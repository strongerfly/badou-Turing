import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('grayHist')
    plt.xlabel("Bins")
    plt.ylabel("hist")
    plt.plot(hist)
    plt.show()


def color_hist(img):
    channels = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("colorHIst")
    plt.xlabel("Bins")
    plt.ylabel("hist")
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def histogram_equal(gray, nbr_bins=256):
    pix_array = np.zeros(256, dtype=np.int)
    height, width = gray.shape
    # 获取灰度级和个数
    for h in range(height):
        for w in range(width):
            gray_level = gray[h, w]
            pix_array[gray_level] += 1
    # pix_sort = np.argsort(pix_array)
    print(pix_array)
    equal_map = np.zeros(256, dtype=np.int)
    sum = 0
    for i in range(len(pix_array)):
        temp = pix_array[i] / (width * height)
        sum += temp
        equal_pix = int(sum * 256 - 1)
        equal_map[i] = equal_pix
    print(equal_map)
    new_img = np.zeros_like(gray)
    for h2 in range(height):
        for w2 in range(width):
            new_img[h2, w2] = equal_map[gray[h2, w2]]

    return new_img


read = cv2.imread('../img/lenna.png', 0)
# gray = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
gray_hist(read)
# color_hist(read)
equal_img = histogram_equal(read)
cv2.imshow('equal', equal_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
gray_hist(equal_img)