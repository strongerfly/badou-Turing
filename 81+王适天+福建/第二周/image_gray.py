#    @author Created by Genius_Tian

#    @Date 2021/6/27

#    @Description AI学习第二周作业
import cv2
import numpy as np
import time


# 灰度化
def rgb2gray(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    empty_img = np.zeros((h, w), img.dtype)
    for i in range(h):
        for j in range(w):
            empty_img[i, j] = (img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3).astype(np.uint8)
    return empty_img


# numpy向量优化
def my_rgb2gray(img_path):
    img = cv2.imread(img_path)
    factors = np.array([0.11, 0.59, 0.3], dtype=np.float32)
    return np.sum(factors * img, axis=2).astype(np.uint8)


if __name__ == "__main__":
    start = time.time()
    gray = rgb2gray("lenna.png")
    end1 = time.time()
    rgb_gray = my_rgb2gray("lenna.png")
    end2 = time.time()
    print("循环耗时%.3f s,numpy向量加速耗时%.3f s" % (end1 - start, end2 - end1))
    print(gray - rgb_gray)
    cv2.imshow("循环", gray)
    cv2.imshow("numpy", gray)
    cv2.waitKey(0)
