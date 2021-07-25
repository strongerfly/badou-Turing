"""透视变换"""
import cv2


def warpMatrix_with_cv(src, src_point, dst_point, dst_size):
    """
    使用cv 进行透视变换
    :param src: 输入原图像
    :param src_point: 原图像需要矫正的点===> 4个像素点
    :param dst_point: 原图像映射到矫正后的4个点 ====> 4个像素点
    :param dst_size: 输出图像尺寸
    :return: 透视变换后的图像
    """
    # 1、根据src_point和dst_point算出warpMatrix公式
    """
    cv2.getPerspectiveTransform(src, dst)
    :param src：源图像中待测矩形的四点坐标
    :param dst：目标图像中矩形的四点坐标
    """
    warpMatrix = cv2.getPerspectiveTransform(src=src_point, dst=dst_point)

    # 2、 获取输出图像
    """
    cv2.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
    src: 输入图像。
    dst: 输出图像。
    M: 3×3 的变换矩阵。
    dsize: 输出图像的大小 (宽和高)。
    flags: 可以选择插值方法 (双线性插值、最近邻插值等，默认是双线性)，还可以设置是否求 M 的逆 (默认 M 是 src->dst 的变换矩阵，设置该选项后 M 是 dst->src 的变换矩阵，也就是 src 要左乘 M 的逆)。
    borderMode: 如何填充输出图像剩余的像素。输入图像经过变换后，输出图像的部分位置可能没有对应的像素值，可以采用填充方法来使输出图像看起来更加自然，可用的方法有填充相同的常数像素值 (默认)、纵向复制边缘像素等。
    borderValue: 当 borderMode 是 “常量填充” 时会用到，指定填充的像素值，默认是 0 (黑色)。
    """
    dst = cv2.warpPerspective(src=src, M=warpMatrix, dsize=dst_size)
    return dst
