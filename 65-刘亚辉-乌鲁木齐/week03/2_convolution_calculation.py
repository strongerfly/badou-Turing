import numpy as np
import cv2
import matplotlib.pyplot as plt


def calcu_padding_size(image_rows, image_cols, core_rows, core_cols, step):
    p_row = int(1/2 *(step*(image_rows-1) - image_rows + core_rows))  # 高
    p_col = int(1/2 *(step*(image_cols-1) - image_cols + core_cols))  # 宽
    return p_row, p_col


def image_padding(image, image_rows, image_cols, p_col, p_row):
    # padding
    padding_rows = np.zeros((p_row, image_cols))
    padding_cols = np.zeros((image_rows+2*p_row, p_col))    # 先加行后，应补齐对应的行数
    # 上下加行
    padding_image = np.vstack((padding_rows, image))
    padding_image = np.vstack((padding_image, padding_rows))
    # 左右加列
    padding_image = np.hstack((padding_cols, padding_image))
    padding_image = np.hstack((padding_image, padding_cols))
    return padding_image


def convolution_valid(padding_image, image_rows, image_cols, core_rows, core_cols, step, p_row, p_col):
    # valid mode
    new_rows = int((image_rows-core_rows)/step + 1)
    new_cols = int((image_cols-core_cols)/step + 1)
    valid_image = np.zeros((new_rows, new_cols))
    for row in range(valid_image.shape[0]):
        for col in range(valid_image.shape[1]):
            # 计算卷积核移动坐标
            x = row*step + p_row  # 加偏移值
            y = col*step + p_col
            # 灰度图中与卷积核对应的矩阵
            # core_image = padding_image[row: row+2*core_radius+1, col: col+2*core_radius+1]
            core_image = padding_image[x: x+core_rows, y: y+core_cols]
            # 卷积运算
            convolution_sum = (core_image * core).sum()
            # 运算结果即卷积后的新图像像素值
            valid_image[row, col] = convolution_sum
    return valid_image


def convolution_same(padding_image, image_rows, image_cols, core_rows, core_cols, step):
    # same mode(卷积后尺寸与原图相同)
    same_image = np.zeros((image_rows, image_cols))
    for row in range(same_image.shape[0]):
        for col in range(same_image.shape[1]):
            # 计算卷积核移动坐标
            x = row*step
            y = col*step
            # 灰度图中与卷积核对应的矩阵
            core_image = padding_image[x: x+core_rows, y: y+core_cols]
            # 卷积运算
            convolution_sum = (core_image * core).sum()
            # 运算结果即卷积后的新图像像素值
            same_image[row, col] = convolution_sum
    return same_image


def convolution_full(image_gray, image_rows, image_cols, core_rows, core_cols, step):
    # full mode
    # 对原图进行padding(padding 两圈)
    padding_image = image_padding(image_gray, image_rows, image_cols, p_row=2, p_col=2)

    full_image = np.zeros((image_rows+2, image_cols+2))
    for row in range(full_image.shape[0]):
        for col in range(full_image.shape[1]):
            # 计算卷积核移动坐标
            x = row*step
            y = col*step
            # 灰度图中与卷积核对应的矩阵
            core_image = padding_image[x: x+core_rows, y: y+core_cols]
            # 卷积运算
            convolution_sum = (core_image * core).sum()
            # 运算结果即卷积后的新图像像素值
            full_image[row, col] = convolution_sum
    return full_image

def convolution_calculation(image, core, core_name, step):
    # 转成灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(221).set_title("GRAY")
    plt.imshow(image_gray, cmap='gray')
    image_rows, image_cols = image_gray.shape

    # 获取卷积核尺寸
    core_rows, core_cols = core.shape
    # 计算核半径
    core_radius = int((core_rows-1)/2)

    ## padding
    # 计算padding尺寸
    p_row, p_col = calcu_padding_size(image_rows, image_cols, core_rows, core_cols, step)
    # 对原图进行padding
    padding_image = image_padding(image_gray, image_rows, image_cols, p_row, p_col)

    # valid 模式卷积图像
    valid_image = convolution_valid(padding_image, image_rows, image_cols, core_rows, core_cols, step, p_row, p_col)
    print('valid_image: ' + str(valid_image.shape))
    # 画出与原图的对比图
    plt.subplot(222).set_title(core_name + '(valid mode)')
    plt.imshow(valid_image.astype(np.uint8), cmap='gray')

    # same 模式卷积图像
    same_image = convolution_same(padding_image, image_rows, image_cols, core_rows, core_cols, step)
    print('same_image: ' + str(same_image.shape))
    # 画出与原图的对比图
    plt.subplot(223).set_title(core_name + '(same mode)')
    plt.imshow(same_image.astype(np.uint8), cmap='gray')

    # full 模式卷积图像
    full_image = convolution_full(image_gray, image_rows, image_cols, core_rows, core_cols, step)
    print('full_image: ' + str(full_image.shape))
    # 画出与原图的对比图
    plt.subplot(224).set_title(core_name + '(full mode)')
    plt.imshow(full_image.astype(np.uint8), cmap='gray')

    # 显示图像
    plt.show()


# 原图
image = cv2.imread('2-265.jpg')

## 平滑卷积滤波
# 定义卷积核
core = np.ones((3, 3)) * (1.0/9)
core_name = 'SEF' # 平滑卷积滤波
# 定义移动窗口的步长
step = 1

convolution_calculation(image, core, core_name, step)


## sobel边缘检测
# image = cv2.imread('sobel_v.png')
# # 定义卷积核
# core = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3) # 纵向
# # core = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3, 3)   # 横向
# core_name = 'Sobel Edge Detection' # sobel边缘检测
# # 定义移动窗口的步长
# step = 1

# convolution_calculation(image, core, core_name, step)


## 高斯平滑
# image = cv2.imread('2-265.jpg')
# # 定义卷积核
# core = np.array([1/16, 2/16, 1/16, 2/16, 2/16, 2/16, 1/16, 2/16, 1/16]).reshape(3, 3)
# core_name = 'Gaussian' # 高斯平滑
# # 定义移动窗口的步长
# step = 1

# convolution_calculation(image, core, core_name, step)
