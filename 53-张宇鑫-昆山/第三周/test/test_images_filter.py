import os
from unittest import TestCase

import cv2
import numpy

from setting_main import IMAGE_DIR
from images_filter import filter_with_cv


class Test(TestCase):
    def test_filter_with_cv(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # 输入彩色图像
        src_bgr = cv2.imread(image_path)
        # 输入灰度图像
        # 边缘提取
        kernel_soble = numpy.array(([1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]))
        # 均值平滑
        kernel_bianyuan = numpy.array(([1/9, 1/9, 1/9],
                                       [1/9, 1/9, 1/9],
                                       [1/9, 1/9, 1/9]),dtype='float32')
        # 高斯平滑
        kernel_gaosi = numpy.array(([1/16, 2/16, 1/16],
                                       [2/16, 1/16, 2/16],
                                       [1/16, 2/16, 1/16]),dtype='float32')
        dst_soble = filter_with_cv(src=src_bgr, kernel=kernel_soble)
        dst_bianyuan = filter_with_cv(src=src_bgr, kernel=kernel_bianyuan)
        dst_gaosi = filter_with_cv(src=src_bgr, kernel=kernel_gaosi)

        cv2.imshow("dst_soble", dst_soble)
        cv2.imshow("dst_bianyuan", dst_bianyuan)
        cv2.imshow("dst_gaosi", dst_gaosi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
