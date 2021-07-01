import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.util import img_as_float


## 1.RBG图像转灰度图像(手动实现+调包实现)
def trans_grey_byhand(image):
    rows, columns, channel = image.shape
    ## 手动实现
    # 构造单通道灰度像素的矩阵
    grey_img_hand = np.zeros((rows, columns), dtype=image.dtype)

    for row in range(rows):
        for col in range(columns):
            # 得到RGB三通道的像素值
            B, G, R = image[row][col]
            # 将每个像素原先的RGB值根据公式转换为单通道的灰度数值
            grey_img_hand[row][col] = 0.3*R + 0.59*G + 0.11*B

    return grey_img_hand


def trans_grey_skimage(image):
    ## 调包
    # 先转成RGB格式，再转灰度
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb2gray(rgb_img)


## 2.RGB图像转黑白二值图像
def trans_binary(image):
    # 将img转成RGB格式
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 先做灰度处理(先转换成浮点数，再做乘法运算)
    # grey_image = (img_rgb / 255.0) @ np.array([0.2125, 0.7154, 0.0721]) # 符合视觉的RGB权重
    grey_image = (img_rgb) @ np.array([0.2125, 0.7154, 0.0721]) # 符合视觉的RGB权重
    # 再进行二值运算
    binary_image = np.where(grey_image >= 0.5, 1.0, 0.0)
    binary_image = (binary_image * 255).astype(image.dtype)
    return binary_image


## 3.最近邻算法
def nerp(image, dst_x, dst_y):
    src_x, src_y, channels = image.shape
    # 构造新图像矩阵
    nerp_image = np.zeros((dst_x, dst_y, channels), dtype=image.dtype)
    # 缩放比例
    scale_x = dst_x / src_x
    scale_y = dst_y / src_y

    for px in range(dst_x):
        for py in range(dst_y):
            # 最近邻算法(根据原图像的像素值更新新图像的像素值)
            x = min(round(px / scale_x), src_x - 1) # 避免越界
            y = min(round(py / scale_y), src_y - 1)
            nerp_image[px, py] = image[x, y]

    return nerp_image


## 4.双线性插值算法
def blerp(image, dst_x, dst_y):
    src_x, src_y, channel = image.shape
    # 构造新图像矩阵
    blerp_image = np.zeros((dst_x, dst_y, channel), dtype=image.dtype)
    # 缩放比例
    scale_x = dst_x / src_x
    scale_y = dst_y / src_y

    for px in range(dst_x):
        for py in range(dst_y):
            # 新坐标值转换到原坐标下(中心对称)
            x = (px + 0.5) / scale_x - 0.5
            y = (py + 0.5) / scale_y - 0.5
            # 找到 (x, y) 的四个邻近点
            x1 = int(np.floor(x))
            x2 = min(x1 + 1, src_x - 1) # 避免越界
            y1 = int(np.floor(y))
            y2 = min(y1 + 1, src_y - 1)
            Q11 = (x1, y1)
            Q21 = (x2, y1)
            Q12 = (x1, y2)
            Q22 = (x2, y2)
            # x方向单线性插值
            r1 = (x2 - x) * image[Q11] + (x - x1) * image[Q21]
            r2 = (x2 - x) * image[Q12] + (x - x1) * image[Q22]
            # y方向单线性插值
            blerp_image[px, py] = (y2 - y) * r1 + (y - y1) * r2

    # 填充黑边
    # 最后一行
    if ((blerp_image[-1] == np.zeros((1, dst_y, channel))).all()):
        blerp_image[-1] = blerp_image[-2] # 等于倒数第二行的像素值
    # 最后一列
    if ((blerp_image[:, -1] == np.zeros((dst_x, 1, channel))).all()):
        blerp_image[:, -1] = blerp_image[:, -2]   # 等于倒数第二列的像素值

    return blerp_image


