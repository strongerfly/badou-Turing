"""逻辑层"""
import numpy
from settings.setting_main import LOG


def image_nearest_interpolation(src: numpy.ndarray, dst_W: int, dst_H: int) -> numpy.ndarray:
    """
    The nearest interpolation
    最邻近差值方式
    :param src: 源数字图像
    :param dst_W: 目标图像像素宽度
    :param dst_H: 目标图像像素高度
    :return: dst 通过最邻近差值方式 获得的放大或缩小后的图片
    """
    src_shape = src.shape
    src_W = src_shape[0]
    src_H = src_shape[1]
    src_BGR = src_shape[2]
    LOG.info("原图像宽度: {},高度:{}, 通道数:{}".format(src_W, src_H, src_BGR))
    LOG.info("变更后图像宽度: {},高度:{}, 通道数:{}".format(dst_W, dst_H, src_BGR))
    # 创建一个 目标像素大小的array
    dst = numpy.zeros((dst_W, dst_H, src_BGR), dtype=numpy.uint8)
    for dst_x in range(dst_W):
        # x方向 目标像素相对于 原图像像素位置
        x_ = (dst_x + 0.5) * src_W / dst_W - 0.5
        # 目标坐标小于 0 的 处理成0
        x_ = x_ if x_ > 0 else 0
        # 目标坐标大于 原图像宽度的 处理成 原图像宽度
        x_ = x_ if x_ < src_W - 1 else src_W - 1
        # 四设五入
        x = int(numpy.round(x_, 0))
        for dst_y in range(dst_H):
            # y方向 目标像素相对于 原图像像素位置
            y_ = (dst_y + 0.5) * src_H / dst_H - 0.5
            y_ = y_ if y_ > 0 else 0
            y_ = y_ if y_ < src_H - 1 else src_H - 1
            y = int(numpy.round(y_, 0))
            dst[dst_x, dst_y] = src[x, y]
    return dst


def image_bilinear_interpolation(src: numpy.ndarray, dst_W: int, dst_H: int) -> numpy.ndarray:
    """
    the bilinear interpolation
    双线差值法
    :param src: 原数字图像
    :param dst_W: 目标图像像素宽度
    :param dst_H: 目标图像像素高度
    :return: dst 通过双线差值方式 获得的放大或缩小后的图片
    """
    src_shape = src.shape
    src_W = src_shape[0]
    src_H = src_shape[1]
    src_BGR = src_shape[2]
    LOG.info("原图像宽度: {},高度:{}, 通道数:{}".format(src_W, src_H, src_BGR))
    LOG.info("变更后图像宽度: {},高度:{}, 通道数:{}".format(dst_W, dst_H, src_BGR))
    # 创建一个 目标像素大小的array
    dst = numpy.zeros((dst_W, dst_H, src_BGR), dtype=numpy.uint8)
    for dst_x in range(dst_W):
        # x方向 目标像素相对于 原图像像素位置
        x_ = (dst_x + 0.5) * src_W / dst_W - 0.5
        # 目标坐标小于 0 的 处理成0
        x_ = x_ if x_ > 0 else 0
        # 目标坐标大于 原图像宽度的 处理成 原图像宽度
        # 为什么是1.1 ???  ===>  对于 720*720 的像素 最后几个像素可能是 719.xxx,719.xxx 最近四个像素为 719,719 719,720 720,719 720,720 超出了原图像的范围
        x_ = x_ if x_ < src_W - 1.1 else src_W - 1.1
        # 取整数
        i = int(x_ // 1)
        # 取余数
        u = x_ % 1
        for dst_y in range(dst_H):
            # y方向 目标像素相对于 原图像像素位置
            y_ = (dst_y + 0.5) * src_H / dst_H - 0.5
            y_ = y_ if y_ > 0 else 0
            y_ = y_ if y_ < src_H - 1.1 else src_H - 1.1
            # 取整数
            j = int(y_ // 1)
            # 取余数
            v = y_ % 1
            dst[dst_x, dst_y] = (1 - u) * (1 - v) * src[i, j] + (1 - u) * v * src[i, j + 1] + u * (1 - v) * src[
                i + 1, j] + u * v * src[i + 1, j + 1]
    return dst
