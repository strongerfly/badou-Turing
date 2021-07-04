#    @author Created by Genius_Tian

#    @Date 2021/7/4

#    @Description 图像直方图均衡化
import cv2
import numpy as np
from histogram import *


def histogram_equalization(image):
    h, w = image.shape
    copy = image.copy()
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    init = 0
    for i in range(256):
        init += (hist[i][0] / (h * w))
        x_index, y_index = np.where(copy == i)
        copy[x_index, y_index] = np.around(init * 256 - 1)
    return copy.astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread(r"../resources/lenna.png", 0)
    equalization = histogram_equalization(image)
    print(image)
    # print(equalization)
    gray_histogram2(image)
    # gray_histogram2(equalization)
    cv2.imshow("origin", image)
    cv2.imshow("e", equalization)

    cv2.waitKey(0)
