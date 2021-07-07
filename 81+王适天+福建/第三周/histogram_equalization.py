#    @author Created by Genius_Tian

#    @Date 2021/7/4

#    @Description 图像直方图均衡化
import cv2
import numpy as np


def histogram_equalization(image):
    h, w = image.shape
    hist_input = cv2.calcHist([image], [0], None, [256], [0, 256])
    empty = np.zeros(image.shape, image.dtype)
    init = 0
    for i in range(256):
        init += hist_input[i][0]
        y_index, x_index = np.where(image == i)
        v = np.around(init * 256 / (h * w) - 1)
        if v > 0:
            empty[y_index, x_index] = v
    return empty.astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread(r"../resources/lenna.png", 0)
    equalization = histogram_equalization(image)
    cv2.imshow("origin", image)
    cv2.imshow("e", equalization)
    cv2.waitKey(0)
