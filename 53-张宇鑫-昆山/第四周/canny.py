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
    abx_x = cv2.convertScaleAbs(src_sobel_X)
    abx_y = cv2.convertScaleAbs(src_sobel_Y)
    sobel_result = cv2.addWeighted(abx_x, 0.5, abx_y, 0.5, 0)
    sobel_float_result = sobel_result.astype(numpy.float32)
    # 将CV_16S型的输出图像转变成float32型的图像
    sobel_float_X = src_sobel_X.astype(numpy.float32)
    sobel_float_Y = src_sobel_Y.astype(numpy.float32)
    # 将sobel_float_X等于0的像素转换为一个极小值,防止算梯度的时候除数为0的情况
    sobel_float_X[sobel_float_X == 0] = 0.0000001
    # 计算斜率
    # 4.对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
    '''
    设中心点坐标为(i,j) 则附近8个点个表示为
    i-1,j-1  i-1,j  i-1,j+1
     i,j-1    i,j    i,j+1
    i+1,j-1  i+1,j  i+1,j+1 
    插值法公式 y = (x1-x)/(x1,x0)*y0 + (x-x0)/(x1,x0)*y1
            由于 x1=x必然等于1 所以公式转化为 (x1-x)*y0 + (x-x0)*y1
    '''
    gradient = sobel_float_Y / sobel_float_X
    for i in range(1, sobel_float_result.shape[0] - 1):
        for j in range(1, sobel_float_result.shape[1] - 1):
            is_max_num = True
            if 1 >= gradient[i, j] > 0:
                # 斜率大于0小于1 ===> 0< 角度 <= 45°  插值算法中 x1-x = 1-gradient[i, j] x-x0=gradient[i, j]
                # 插值法 4个点 为 (i-1,j+1),(i,j+1)  和 (i,j-1),(i+1,j-1)
                p1 = 1 - gradient[i, j]
                p2 = gradient[i, j]
                y01 = sobel_float_result[i - 1, j + 1]
                y02 = sobel_float_result[i, j + 1]
                y11 = sobel_float_result[i, j - 1]
                y12 = sobel_float_result[i + 1, j - 1]
                sobel_pix = sobel_float_result[i, j]
                _, _, is_max_num = gradient_result(p1, p2, y01, y02, y11, y12, sobel_pix)

            elif gradient[i, j] > 1:
                # 斜率大于1 ===> 45< 角度 <= 90°  插值算法中 x1-x = 1-1/gradient[i, j] x-x0=1/gradient[i, j]
                # 插值法 4个点 为 (i-1,j),(i-1,j+1)  和 (i+1,j-1),(i+1,j)
                p1 = 1 - 1 / gradient[i, j]
                p2 = 1 / gradient[i, j]
                y01 = sobel_float_result[i - 1, j]
                y02 = sobel_float_result[i - 1, j + 1]
                y11 = sobel_float_result[i + 1, j - 1]
                y12 = sobel_float_result[i + 1, j]
                sobel_pix = sobel_float_result[i, j]
                _, _, is_max_num = gradient_result(p1, p2, y01, y02, y11, y12, sobel_pix)
            elif 0 >= gradient[i, j] > -1:
                # 斜率小于0并大于-1 ===> -45< 角度 <= 0  插值算法中 x1-x = 1-gradient[i, j] x-x0=gradient[i, j]
                # 插值法 4个点 为 (i-1,j-1),(i,j-1)  和 (i,j+1),(i+1,j+1)
                p1 = 1 - gradient[i, j]
                p2 = gradient[i, j]
                y01 = sobel_float_result[i - 1, j - 1]
                y02 = sobel_float_result[i, j - 1]
                y11 = sobel_float_result[i, j + 1]
                y12 = sobel_float_result[i + 1, j + 1]
                sobel_pix = sobel_float_result[i, j]
                _, _, is_max_num = gradient_result(p1, p2, y01, y02, y11, y12, sobel_pix)
            elif gradient[i, j] <= -1:
                # 斜率小于等于-1 ===>  角度 <= -45  插值算法中 x1-x = 1-1/gradient[i, j] x-x0=1/gradient[i, j]
                # 插值法 4个点 为 (i-1,j-1),(i-1,j)  和 (i+1,j),(i+1,j+1)
                p1 = 1 - 1 / gradient[i, j]
                p2 = 1 / gradient[i, j]
                y01 = sobel_float_result[i - 1, j - 1]
                y02 = sobel_float_result[i - 1, j]
                y11 = sobel_float_result[i + 1, j]
                y12 = sobel_float_result[i + 1, j + 1]
                sobel_pix = sobel_float_result[i, j]
                _, _, is_max_num = gradient_result(p1, p2, y01, y02, y11, y12, sobel_pix)

            if not is_max_num:
                # 如果不是极大值抑制为0
                sobel_float_result[i, j] = 0.
    # 将非极大值抑制后的图像转换为uint8
    unit_result = sobel_float_result.astype(numpy.uint8)
    # 5.用双阈值算法检测和连接边缘
    '''
    1、先获取强边缘像素下标
    2、在强边缘邻近八点看是否有中边缘像素 若有 把他认为强边缘像素
    3、把中边缘像素装换到强边缘后 添加到步骤1
    '''
    # 获取强边缘下标
    npWhere = numpy.where(unit_result >= high_boundary)
    # 将强边缘像素置为255
    unit_result[unit_result >= high_boundary] = 255
    # 将弱边缘像素置为0
    unit_result[unit_result <= lower_boundary] = 0
    # 获取原图中强边缘个数
    high_boundary_num = len(npWhere[0])
    num = 0
    m = []
    for i in range(len(npWhere[0])):
        m.append([npWhere[0][i],npWhere[1][i]])
    while high_boundary_num != 0:
        # 强像素坐标
        high_boundary_X = m[num][0]
        high_boundary_y = m[num][1]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    # 屏蔽中心位置
                    continue
                # 确保周围点能在图像中取到
                if 0 <= high_boundary_X + i < unit_result.shape[0]:
                    if 0 <= high_boundary_y + j < unit_result.shape[1]:
                        # 判断周边像素是否存在中边缘值
                        if lower_boundary<unit_result[high_boundary_X + i,high_boundary_y + j]<high_boundary:
                            # 赋值为强边缘
                            unit_result[high_boundary_X + i,high_boundary_y + j] = 255
                            m.append([high_boundary_X + i, high_boundary_y + j])
                            high_boundary_num += 1
        num +=1
        high_boundary_num -=1
    unit_result[unit_result!=255] = 0

    dst = unit_result
    return dst


def gradient_result(p1, p2, y01, y02, y11, y12, sobel_pix) -> object:
    """
    梯度插值算法
    :param p1: 相当于上面函数中的x1-x
    :param p2: 相当于上面函数中的x-x0
    :param y01: 两个邻近点的像素值1
    :param y02: 两个邻近点的像素值2
    :param y11: 两个邻近点的像素值3
    :param y12: 两个邻近点的像素值4
    :param sobel_pix: 原图像素
    :return: g1,g2 插值算法后输出
            is_max_num: False 表示 不是极大值
                        True 表示 是极大值
    """
    g1 = p1 * y01 + p2 * y02
    g2 = p1 * y11 + p2 * y12
    is_max_num = True
    if sobel_pix < g1 or sobel_pix < g2:
        is_max_num = False
    return g1, g2, is_max_num


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
