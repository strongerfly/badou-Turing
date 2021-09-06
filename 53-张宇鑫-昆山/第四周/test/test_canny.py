"""canny测试"""
import os
from unittest import TestCase

import cv2

from settings.setting_main import IMAGE_DIR
from canny import canny_with_myself, canny_with_cv


class Test(TestCase):
    def test_canny_with_myself(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # 输入彩色图像
        src_bgr = cv2.imread(image_path)
        dst = canny_with_myself(src=src_bgr, lower_boundary=100, high_boundary=300)

        cv2.imshow("src_bgr", src_bgr)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_canny_with_cv(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # 输入彩色图像
        src_bgr = cv2.imread(image_path)
        dst = canny_with_cv(src=src_bgr, lower_boundary=100, high_boundary=300)
        cv2.imshow("src_bgr", src_bgr)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


