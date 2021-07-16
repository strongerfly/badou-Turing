#    @author Created by Genius_Tian

#    @Date 2021/7/4

#    @Description 图像直方图均衡化
import cv2
import numpy as np


def histogram_equalization(image):
    h, w = image.shape[0], image.shape[1]
    split = cv2.split(image)
    t = []
    for i in split:
        hist_input = cv2.calcHist([i], [0], None, [256], [0, 256])
        empty = np.zeros(i.shape, i.dtype)
        init = 0
        for pi in range(256):
            init += hist_input[pi][0]
            y_index, x_index = np.where(i == pi)
            v = np.around(init * 256 / (h * w) - 1)
            if v > 0:
                empty[y_index, x_index] = v
        t.append(empty)
    return cv2.merge(t).astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread(r"../resources/lenna.png", 0)
    equalization = histogram_equalization(image)
    cv2.imshow("origin", image)
    cv2.imshow("e", equalization)
    cv2.waitKey(0)
