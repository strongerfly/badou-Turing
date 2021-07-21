from unittest import TestCase
import cv2
import numpy

from setting_main import IMAGE_DIR, LOG
from PCA import PCA_sklearn, PCA_myself,aaaa


class Test(TestCase):
    def test_pca_sklearn(self):
        src = numpy.array([
            [-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]
        ])
        dst = PCA_sklearn(src,2)
        LOG.info('dst: {}'.format(dst))

    def test_pca_myself(self):
        src = numpy.array([
            [-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]
        ])
        dst1 =PCA_myself(src,2)
        LOG.info('dst1: {}'.format(dst1))

