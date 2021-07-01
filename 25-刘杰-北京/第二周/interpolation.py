'''
@Project ：badou-Turing
@File    ：nearist_interpolation.py
@Author  ：luigi
@Date    ：2021/6/27 下午3:45
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from nearest import nearest_sample
from bilinear import bilinear_sample
from cvt_gray import bgr2gray

def interpolate(input, size, offset=0.5, mode: str = 'nearest'):
    """Down/up samples the input to either the given size

    :param input: 输入图像
    :type input: np.array(np.uint8)
    :param size: 目标图像的长宽，比如(3,4)表示目标图像为3行4列
    :type size: tuple
    :param offset: 偏移，默认为0.5，表示几何中心重合
    :type offset: float
    :param mode: 图像采样方法：最邻近插值法和双线性插值法
    :type mode: string
    :return: 插值采样后的目标图像
    :rtype: np.array(np.uint8)
    """
    source_height, source_width = input.shape
    target_height, target_width = size
    target_grid = np.indices(size)
    source_x_indices = (target_grid[1] + offset) * (source_width / target_width) - offset
    source_y_indices = (target_grid[0] + offset) * (source_height / target_height) - offset

    if mode == "nearest":
        return nearest_sample(input,source_x_indices,source_y_indices)
    if mode == "bilinear":
        return bilinear_sample(input,source_x_indices,source_y_indices)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--path",required=True,help="path for input image")
    ap.add_argument("-he","--height",required=True,help="height for target image")
    ap.add_argument("-w","--width",required=True,help="width for target image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = bgr2gray(image)
    try:
        height = int(args["height"])
        width = int(args["width"])
    except Exception:
        raise argparse.ArgumentTypeError("must enter integer for height and width parameter")
    size = (height,width)

    target_bilinear = interpolate(gray,size,mode='bilinear')
    target_nearest = interpolate(gray,size)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    fig.tight_layout(pad=2.0) #subplot间距
    ax1.title.set_text('原始灰度图像')
    ax1.imshow(gray,cmap='gray')
    ax2.title.set_text('双线性插值')
    ax2.imshow(target_bilinear,cmap='gray')
    ax3.title.set_text('最邻近插值')
    ax3.imshow(target_nearest,cmap='gray')

    plt.show()

if __name__ == '__main__':
    main()
