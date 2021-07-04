#    @author Created by Genius_Tian

#    @Date 2021/7/4

#    @Description 图像直方图均衡化
import cv2
import numpy as np


def histogram_equalization(image):
    h, w = image.shape
    copy = image.copy()
    hist_input = cv2.calcHist([image], [0], None, [256], [0, 256])
    init = 0
    hist_out = []
    for i in range(256):
        init += (hist_input[i][0] / (h * w))
        print(i, init)
        hist_out.append(np.around(max(0, init * 256 - 1)))
    for y in range(h):
        for x in range(w):
            copy[y, x] = hist_out[copy[y, x]]
    return copy.astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread(r"../resources/lenna.png", 0)
    equalization = histogram_equalization(image)
    # print(image)
    # print(equalization)
    hist = cv2.equalizeHist(image)
    # print(hist - equalization)
    cv2.imshow("opencv", hist)
    cv2.imshow("origin", image)
    cv2.imshow("e", equalization)
    cv2.waitKey(0)
