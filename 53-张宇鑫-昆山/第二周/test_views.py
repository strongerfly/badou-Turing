import os
from unittest import TestCase
import cv2
import numpy

from setting_main import IMAGE_DIR, LOG

import views


class Test(TestCase):
    def test_image_interpolation(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # src = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        src = cv2.imread(image_path)
        LOG.info(src.shape)
        dst_W = int(src.shape[0] * 2 - 500)
        dst_H = int(src.shape[1] * 2 - 300)
        dst_nearest = views.image_nearest_interpolation(src, dst_W, dst_H)
        cv2.imshow("src", src)
        cv2.imshow("dst_nearest", dst_nearest)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def test(self):
        # x = []
        # y = []
        # for i in range(13900):
        #     x.append((i + 0.5) * 720 / 13900 - 0.5)
        # for j in range(14100):
        #     y.append((j + 0.5) * 720 / 14100 - 0.5)
        # LOG.info("x: {} ".format(x))
        # LOG.info("y: {} ".format(y))
        a = numpy.array([1, 2, 3])
        b = a + 1
        c = a * 3
        pass

    def test_image_bilinear_interpolation(self):
        image_path = os.path.join(IMAGE_DIR, 'lenna.jpg')
        # src = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        src = cv2.imread(image_path)
        LOG.info(src.shape)
        dst_W = int(src.shape[0] * 2 - 500)
        dst_H = int(src.shape[1] * 2 - 300)
        dst_bilinear = views.image_bilinear_interpolation(src, dst_W, dst_H)
        cv2.imshow("src", src)
        cv2.imshow("dst_bilinear", dst_bilinear)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
