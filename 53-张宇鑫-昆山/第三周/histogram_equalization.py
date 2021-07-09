"""直方图均衡化实现"""
import cv2
import numpy
from settings.setting_main import LOG

def image_histogram_equalization_with_cv(src: numpy.ndarray) -> numpy.ndarray:
    """
    方法1：使用cv方法直线直方图均值化
    :param src: 输入原始图像
    :return: 直方图均值化后的图像
    """
    src_shape = src.shape  # 数组形状
    src_ndim = src.ndim   # 数据维度
    result_image = []
    dst = None
    if src_ndim.__eq__(2):
        # 2维数据  灰色图像
        dst = cv2.equalizeHist(src)
        pass
    elif src_ndim.__eq__(3):
        # 3维数据  彩色图像
        (b,g,r) = cv2.split(src)
        eq_b = cv2.equalizeHist(b)
        eq_g = cv2.equalizeHist(g)
        eq_r = cv2.equalizeHist(r)
        dst = cv2.merge((eq_b,eq_g,eq_r))
    return dst

