import os
from unittest import TestCase

import cv2
import numpy

from settings.setting_main import IMAGE_DIR
from warpMatrix import warpMatrix_with_cv


class Test(TestCase):
    def test_warp_matrix_with_cv(self):

        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # 输入彩色图像
        src_bgr = cv2.imread(image_path)
        src_point = numpy.float32([[0,0],[0,700],[700,0],[700,700]])
        dst_point = numpy.float32([[0,500],[0,1000],[700,0],[700,500]])
        dst_size = (1001,1001)
        dst = warpMatrix_with_cv(src_bgr, src_point, dst_point, dst_size)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
