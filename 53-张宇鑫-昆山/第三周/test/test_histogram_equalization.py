import os
from unittest import TestCase

import cv2
from matplotlib import pyplot as plt

from setting_main import IMAGE_DIR

from histogram_equalization import image_histogram_equalization_with_cv


class Test(TestCase):
    def test_image_histogram_equalization_with_cv(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # 输入彩色图像
        src_bgr = cv2.imread(image_path)
        # 输入灰度图像
        src_gay = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        # 均衡化后的彩色图像
        eq_bgr = image_histogram_equalization_with_cv(src_bgr)
        # 均衡化后的灰度图像
        eq_gay = image_histogram_equalization_with_cv(src_gay)
        # 彩色原图像 b 直方图·

        (src_bgr_b, src_bgr_g, src_bgr_r) = cv2.split(src_bgr)
        hist_bgr_b = cv2.calcHist(images=[src_bgr_b], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # 彩色原图像 g 直方图
        hist_bgr_g = cv2.calcHist(images=[src_bgr_g], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # 彩色原图像 r 直方图
        hist_bgr_r = cv2.calcHist(images=[src_bgr_r], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

        (eq_bgr_b, eq_bgr_g, eq_bgr_r) = cv2.split(eq_bgr)
        # 均衡化后 彩色 b 直方图
        hist_eq_bgr_b = cv2.calcHist(images=[eq_bgr_b], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # 均衡化后 彩色 g 直方图
        hist_eq_bgr_g = cv2.calcHist(images=[eq_bgr_g], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # 均衡化后 彩色 r 直方图
        hist_eq_bgr_r = cv2.calcHist(images=[eq_bgr_r], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

        # 原灰度 图像直方图
        hist_gay = cv2.calcHist(images=[src_gay], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        # 均值化后 灰度图像直方图
        hist_eq_gay = cv2.calcHist(images=[eq_gay], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

        plt_title = ['hist_bgr_b', 'hist_bgr_g', 'hist_bgr_r', 'hist_eq_bgr_b', 'hist_eq_bgr_g', 'hist_eq_bgr_r',
                     'hist_gay', 'hist_eq_gay']
        for i, j in enumerate([
            hist_bgr_b, hist_bgr_g, hist_bgr_r, hist_eq_bgr_b, hist_eq_bgr_g, hist_eq_bgr_r, hist_gay, hist_eq_gay]):
            # 显示 原彩色图像 b 直方图
            plt.figure()  # 新建图像
            plt.title(plt_title[i])  # 标题
            plt.xlabel('Bins')  # x轴标签
            plt.ylabel('Pixels')  # y轴标签
            plt.plot(j)
            plt.xlim([0, 256])  # 设置x轴方位
            plt.show()  # 显示
        cv2.imshow("src_bgr", src_bgr)

        cv2.imshow("eq_bgr", eq_bgr)
        cv2.imshow("src_gay", src_gay)
        cv2.imshow("eq_gay", eq_gay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
