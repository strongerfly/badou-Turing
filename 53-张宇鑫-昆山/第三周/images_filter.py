"""卷积操作"""
import cv2
import numpy


def filter_with_cv(src:numpy.ndarray, kernel:numpy.ndarray) -> numpy.ndarray:
    """
    卷积操作
    :param src:  输入原图像
    :param kernel: 卷积核
    :param paddingType: 卷积方式 1：full  2：same  3：valid
    :return:  输出图像
    """
    res = cv2.filter2D(src, -1, kernel)
    return res
