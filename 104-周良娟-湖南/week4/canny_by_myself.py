import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# canny 的 5个步骤
# 1. 灰度化
def color_gray(img):
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img_i = img[i, j]
            img_gray[i, j] = int(img_i[0] * 0.11 + img_i[1] * 0.59 + img_i[2] * 0.3)
    return img_gray

    pass


# 2. 高斯核函数, 老师的参考
def Gauss_2d_kernel1(sigma):
    '''
    # https://blog.csdn.net/blogshinelee/article/details/82734769
    # https://www.jianshu.com/p/d21a33a7901a
    # σ=0.3×((ksize−1)×0.5−1)+0.8
    :param sigma:
    :return:
    '''
    ksize = int(6 * sigma + 1)
    gauss_kernel = np.zeros((ksize, ksize))  # 核函数初始化
    if ksize // 2 == 0:  # 偶数kernel，+1，变成奇数
        ksize += 1
    temp = [i - ksize for i in range(ksize)]  # 确定变量的取值
    val1 = 1 / (2 * math.pi * sigma ** 2)
    val2 = -1 / (2 * sigma ** 2)
    for i in range(ksize):
        for j in range(ksize):
            gauss_kernel[i, j] = val1 * math.exp(val2 * (i ** 2 + y ** 2))
    return gauss_kernel


def Gauss_2d_kernel1(sigma):
    '''
    # https://blog.csdn.net/blogshinelee/article/details/82734769
    # σ=0.3×((ksize−1)×0.5−1)+0.8
    :param sigma:
    :return:
    '''
    ksize = int(np.round(6 * sigma + 1))
    if ksize % 2 == 0:  # 偶数kernel，+1，变成奇数
        ksize += 1
    gauss_kernel = np.zeros((ksize, ksize))  # 核函数初始化
    temp = [i - ksize // 2 for i in range(ksize)]  # 确定变量的取值
    val1 = 1 / (2 * math.pi * sigma ** 2)
    val2 = -1 / (2 * sigma ** 2)
    for i in range(ksize):
        for j in range(ksize):
            gauss_kernel[i, j] = val1 * math.exp(val2 * (temp[i] ** 2 + temp[j] ** 2))
    return gauss_kernel / np.sum(gauss_kernel)  # gauss_kernel.sum()


def Gauss_2d_kernel2(ksize, sigma=0):
    '''# σ=0.3×((ksize−1)×0.5−1)+0.8'''
    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    gauss_kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    val1 = 1 / (2 * math.pi * sigma ** 2)
    val2 = -1 / (2 * sigma ** 2)
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            gauss_kernel[i, j] = val1 * math.exp(val2 * (i ** 2 + y ** 2))
    return gauss_kernel / gauss_kernel.sum()


# 高斯核生成函数  网上
def creat_gauss_kernel(kernel_size=3, sigma=1, k=1):
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    X = np.linspace(-k, k, kernel_size)
    Y = np.linspace(-k, k, kernel_size)
    x, y = np.meshgrid(X, Y)
    x0 = 0
    y0 = 0
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return gauss


# 生成gauss_kernel，同时进行blur
def GaussianBlur(img, ksize, sigma=0):
    if sigma == 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    gauss_kernel = np.zeros((ksize, ksize))
    center = ksize // 2
    val1 = 1 / (2 * math.pi * sigma ** 2)
    val2 = -1 / (2 * sigma ** 2)
    for i in range(ksize):
        for j in range(ksize):
            x = i - center
            y = j - center
            gauss_kernel[i, j] = val1 * math.exp(val2 * (x ** 2 + y ** 2))

    # padding
    dx, dy = img.shape
    radius = ksize // 2
    img_new = np.zeros([dx, dy])
    img_pad = np.pad(img, ((radius, radius), (radius, radius)), 'constant')  # 备注1
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + ksize, j:j + ksize] * gauss_kernel)
    return img_new.astype(np.uint8)


# 3. 边缘提取  求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
def filter(img, option='sobel'):
    '''
    :param option:
    :return:
    '''
    if option == 'sobel':
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    if option == 'prewitt':
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    dx, dy = img.shape
    radius = 1
    angle = np.zeros([dx, dy])  # 计算斜率
    img_gradient = np.zeros([dx, dy])
    img_pad = np.pad(img, ((radius, radius), (radius, radius)), 'constant')  # 备注1
    for i in range(dx):
        for j in range(dy):
            x = np.sum(img_pad[i:i + 3, j:j + 3] * kernel_x)
            y = np.sum(img_pad[i:i + 3, j:j + 3] * kernel_y)
            img_gradient[i, j] = np.sqrt(x ** 2 + y ** 2)
            if x == 0:
                x = 1e-6
            angle[i, j] = y / x
    return angle, img_gradient.astype(np.uint8)


# 4. 非最大值抑制原理
def non_maximum_suppresed1(img_gradient, angle):
    '''
    :param img_gray:
    :return:
    '''
    dx, dy = img_gradient.shape
    img_nms = np.zeros([dx, dy])
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            temp = img_gradient[i - 1:i + 2, j - 1:j + 2]
            # [00 01 02
            # 10 11 12
            # 20 21 22]
            # 注意斜率angle与图像的焦点位置可以求出来，y = tan(\theta) x, 与y-yo = (y1 - y0) *x 的焦点
            # 插值，插值，https://cloud.tencent.com/developer/article/1408073
            # angle => 1, weight = 1/ angle,angle <= 1, weight = angle,
            # weight = |gx| / |gy| or |gy| / |gx|
            # dTemp1 = weight*g1 + (1-weight)*g2      (g1 - g2)*weight + g2
            # dTemp2 = weight*g3 + (1-weight)*g4
            if abs(angle[i, j]) >= 1:
                weight = 1 / angle[i, j]
                g2 = temp[0, 1]
                g4 = temp[2, 1]
                if angle[i, j] <= -1:
                    # weight = 1 / angle
                    # g1 g2
                    #     c
                    #     g4  g3
                    g1 = temp[0, 0]
                    g3 = temp[2, 2]
                else:
                    # weight = 1 / angle
                    #     g2  g1
                    #     c
                    # g3  g4
                    g1 = temp[0, 2]
                    g3 = temp[2, 0]
            else:
                weight = angle[i, j]
                g2 = temp[1, 0]
                g4 = temp[1, 2]
                if angle[i, j] > -1:
                    # weight = angle
                    # g1
                    # g2  c  g4
                    #        g3
                    g1 = temp[0, 0]
                    g3 = temp[2, 2]
                else:
                    # weight =  angle
                    #         g3
                    # g2  c   g4
                    # g1
                    g1 = temp[2, 0]
                    g3 = temp[0, 2]
            # 对梯度进行插值
            g1,g2,g3,g4 = int(g1),int(g2),int(g3),int(g4)
            grad1 = weight * (g1 - g2) + g2
            grad2 = weight * (g3 - g4) + g4
            if (grad1 <= img_gradient[i, j] and grad2 <= img_gradient[i, j]):
                img_nms[i, j] = img_gradient[i, j]
    return img_nms.astype(np.uint8)


# 5. 双抑制算法
def double_threshold_check(img_nms, min_thre = None, max_thre=None):
    '''
    :param img:
    :param min_value:
    :param max_value:
    :return:
    '''
    h,w = img_nms.shape
    DT = np.zeros([h,w])
    zhan = []
    if not min_thre:
        min_thre = 0.5 * img_nms.mean()
    if not max_thre:
        max_thre = 3 * min_thre
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img_nms[i,j] <= min_thre:
                DT[i,j] = 0
            elif img_nms[i, j] >= max_thre:
                DT[i,j] = 255
                zhan.append([i,j])
            elif (img_nms[i-1, [j-1, j, j+1]] > max_thre).any() or \
                (img_nms[i+1, [j-1, j, j+1]] > max_thre).any() \
                or (img_nms[i, [j -1, j+1]] > max_thre).any():
                DT[i,j] = 255
    return DT.astype(np.uint8)

if __name__ == "__main__":
    path = 'lenna.png'
    img = cv2.imread(path, 1)
    img = color_gray(img)    # 1.灰度灰度化
    img = GaussianBlur(img, 5, sigma=0)  # 2. 高斯平滑
    angle_sobel, img_gradient_sobel = filter(img, option='sobel')   # 3. 滤波
    img_nms = non_maximum_suppresed1(img_gradient_sobel, angle_sobel)   # 4. 非极大值抑制
    DT = double_threshold_check(img_nms)     # 5. 双阈值检测
    cv2.imshow('DT_img', DT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path)
    # print(img)
    img = color_gray(img)  # 1. 灰度化
    # print(img)
    cv2.imshow('gray_lenna', img)
    cv2.waitKey(0)
    # cv2.destroyWindow('gray_lenna')
    # 调用 cv2的gauss_kernel GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    cv2_gau_kernel = cv2.GaussianBlur(img, (5, 5), 0,
                                      0)  # https://blog.csdn.net/weixin_40922285/article/details/102801633 各种滤波
    print(cv2_gau_kernel, 'cv2_gau_kernel')
    cv2.imshow('cv2_gauss_lenna', cv2_gau_kernel)
    cv2.waitKey(0)
    # 自己写的GaussianBlur
    img_gau = GaussianBlur(img, 5, sigma=0)
    print(img_gau, 'img_gau')
    cv2.imshow('gauss_lenna', img_gau)
    cv2.waitKey(0)
    # sobel 提取边缘
    angle_sobel, img_gradient_sobel = filter(img_gau, option='sobel')
    cv2.imshow('gradient_sobel_lenna', img_gradient_sobel)
    cv2.waitKey(0)
    # prewitt 提取边缘
    angle_prewitt, img_gradient_prewitt = filter(img_gau, option='prewitt')
    cv2.imshow('gradient_prewitt_lenna', img_gradient_prewitt)
    cv2.waitKey(0)
    # 非极大值抑制
    img_nms = non_maximum_suppresed1(img_gradient_sobel, angle_sobel)
    cv2.imshow('img_nms', img_nms)
    cv2.waitKey(0)
    # 双阈值法 检测
    DT = double_threshold_check(img_nms)
    cv2.imshow('DT_img', DT)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

# 备注1
'''
np.pad?
# np.pad(array, pad_width, mode='constant', **kwargs)

array, 数组
pad_width, 周围填充的宽度
mode='constant' 填充的模式，默认为常数

a = np.arange(6)
a = a.reshape((2, 3))
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
a1 = np.pad(a, 2, pad_with)
print(a1)
a2 = np.pad(a, 2, pad_with, padder=100)
print(a2, 'a2')

np.pad(a, (2, 3), 'constant', constant_values=(4, 6))

np.pad(a, ((tmp, tmp), (tmp + 1, tmp)), 'constant')  # 边缘填补
'''

'''    gau_kernel1 = Gauss_2d_kernel1(sigma=0.5)
    print(gau_kernel1, 'gau_ker111')
    gau_kernel2 = Gauss_2d_kernel2(ksize=5, sigma=0)
    print(gau_kernel2, 'gau_ker222')
    gau_kernel3 = creat_gauss_kernel(kernel_size=5, sigma=1, k=1)
    print(gau_kernel3, 'gau_ker3')'''
