import numpy
import cv2


def canny_with_myself(src: numpy.ndarray, lower_boundary: int, high_boundary: int) -> numpy.ndarray:
    """
    自己写一个canny 算法
    :param src: 输入原图像
    :param lower_boundary: 阈值低点
    :param high_boundary:  阈值高点
    :return: canny算法过后的输出图像
    1. 对图像进行灰度化
    2. 对图像进行高斯滤波：
       根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
       可以有效滤去理想图像中叠加的高频噪声。
    3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
    4 对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点
      所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
    5 用双阈值算法检测和连接边缘
    """
    # 1. 对图像进行灰度化
    src_gray = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2GRAY)
    # 2. 对图像进行高斯平滑滤波
    '''
    cv自带高斯函数cv2.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    :param src: 输入图像  
    :param ksize: 滤波大小(卷积核)
    :param sigmaX: 卷积核 X方向上的方差 一般设置为0 表示 方差按ksize的大小自动算出
    :param dst: 输出图像 python中不用理会
    :param sigmaY: 卷积核 Y方向上的方差 一般不填或设置为0 表示 和sigmaX参数相同
    :param borderType: 图像边界处理 
    '''
    src_guess = cv2.GaussianBlur(src=src_gray, ksize=(5, 5), sigmaX=0)
    # 3.检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）
    '''
    cv2.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None, delta=None, borderType=None):
    :param src:输入图像
    :param ddepth: 图像深度 -1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
        一般使用cv2.CV_16S。因为OpenCV文档中对Sobel算子的介绍中有这么一句：“in the case of 8-bit input images it will result in truncated derivatives”。
        即Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
        因此要使用16位有符号的数据类型，即cv2.CV_16S
        在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。
    :param dx:表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。 若dx为1 表示图像对x方向进行sobel
    :param dy:通dx 若dy为1 表示图像对y方向进行sobel
    :param ksize: sobel算子大小，必须为整数 奇数
    :param scale: 缩放导数的比例常数，默认情况下没有伸缩系数
    :param delta: 可选增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中
    :param borderType: 图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT
    '''
    src_sobel_X = cv2.Sobel(src_guess, cv2.CV_16S, 1, 0, ksize=3)
    src_sobel_Y = cv2.Sobel(src_guess, cv2.CV_16S, 0, 1, ksize=3)
    '''
    使用convertScaleAbs(src, dst=None, alpha=None, beta=None): 转回uint8形式 将CV_16S型的输出图像转变成CV_8U型的图像。
    dst = |alpha*src+beta|
    :param src:输入图像
    :param alpha:乘数因子
    :param beta:偏移量
    '''
    absX = cv2.convertScaleAbs(src_sobel_X)
    absY = cv2.convertScaleAbs(src_sobel_Y)
    '''
    使用addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None) 实现以不同的权重将两幅图片叠加，对于不同的权重，叠加后的图像会有不同的透明度
    :param src1:输入图像1
    :param alpha:第一个数组的权重
    :param src2:输入图像2
    :param beta:第二个数组的权重值，值为1-alpha
    :param gamma:一个加到权重总和上的标量值，可以理解为加权和后的图像的偏移量
    :param dtype:可选，输出阵列的深度，有默认值-1。当两个输入数组具有相同深度时，这个参数设置为-1（默认值），即等同于src1.depth()。
    dst = alpha*src1+beta*scr2+gamma
    '''
    src_sobel = cv2.addWeighted(src1=absX, alpha=0.5, src2=absY, beta=0.5, gamma=0)

    return src_sobel


def canny_with_cv(src: numpy.ndarray, lower_boundary: int, high_boundary: int) -> numpy.ndarray:
    """
    使用opencv canny 算法
    :param src: 输入原图像
    :param lower_boundary: 阈值低点
    :param high_boundary:  阈值高点
    :return: canny算法过后的输出图像
    1. 对图像进行灰度化
    2. 使用
    """
    '''
    cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None) 进行灰度处理
    :param image:输入图像 必须为灰度图像
    :param threshold1:阈值1（最小值）
    :param threshold2:阈值2（最大值），使用此参数进行明显的边缘检测
    :param edges:图像边缘信息
    :param apertureSize:sobel算子（卷积核）大小
    :param L2gradient:布尔值。
                    True： 使用更精确的L2范数进行计算（即两个方向的导数的平方和再开方）
                    False：使用L1范数（直接将两个方向导数的绝对值相加）
    '''
    dst = cv2.Canny(image=src, threshold1=lower_boundary, threshold2=high_boundary, edges=None, apertureSize=None,
                    L2gradient=None)
    return dst
