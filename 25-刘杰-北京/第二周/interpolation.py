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

class Interpolation():

    def __init__(self, source_image):
        self.source_image = source_image
        self.source_height,self.source_width = self.source_image.shape

    def generate_indices(self,size,offset=0.5):
        """

        :param size: 指定目标图像的长宽，比如(3,4)表示目标图像为3行4列
        :type size: tuple
        :param offset: 偏移，默认为0.5，表示几何中心重合
        :type offset: float
        :return: 返回目标坐标在原坐标系下的虚拟坐标
        :rtype: tuple
        """
        target_height = size[0]
        target_width = size[1]
        target_grid = np.indices(size)
        source_x_indices = (target_grid[1]+offset)*(self.source_width/target_width)-offset
        source_y_indices = (target_grid[0]+offset)*(self.source_height/target_height)-offset
        return (source_x_indices,source_y_indices)


    def nearest_interpolate(self,size):
        """

        :param size: 指定目标图像的像素宽高，比如(3,4)表示目标图像矩阵为3行4列
        :type size: tuple
        :return: 最邻近插值后的图像矩阵
        :rtype: numpy.ndarray
        """
        x, y = self.generate_indices(size)
        x1 = np.rint(x).astype(np.int32)
        y1 = np.rint(y).astype(np.int32)
        return self.source_image[y1,x1]

    def bilinear_interpolate(self, size):
        """

        :param size: 指定目标图像的像素宽高，比如(3,4)表示目标图像矩阵为3行4列
        :type size: tuple
        :return: 双线性插值后的图像矩阵
        :rtype: numpy.ndarray
        """
        x, y = self.generate_indices(size)
        x1 = x.astype(np.int32)
        y1 = y.astype(np.int32)
        x2 = x1 + 1
        y2 = y1 + 1
        # 防止index xxx is out of bounds
        image_with_border = cv2.copyMakeBorder(self.source_image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        # x2[np.where(x2>x1.max())]=x1.max()
        # y2[np.where(y2>y1.max())]=y1.max()

        # 矩阵的行表示的是坐标系的列，矩阵的列表示的是坐标系的行
        z = (y2-y) * (x2-x) * image_with_border[y1, x1] \
            + (y2 - y) * (x - x1) * image_with_border[y1, x2] \
            + (y - y1) * (x2 - x) * image_with_border[y2, x1] \
            + (y - y1) * (x - x1) * image_with_border[y2, x2]

        return z

def cvtGray(image):

    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    gray = b*0.11 + g*0.59 + r*0.3

    return gray

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--path",required=True,help="path for input image")
    ap.add_argument("-he","--height",required=True,help="height for target image")
    ap.add_argument("-w","--width",required=True,help="width for target image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["path"])
    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cvtGray(image)
    try:
        height = int(args["height"])
        width = int(args["width"])
    except Exception:
        raise argparse.ArgumentTypeError("must enter integer for height and width parameter")
    size = (height,width)

    b = Interpolation(gray)
    target_bilinear = b.bilinear_interpolate(size).astype(np.uint8)
    target_nearest = b.nearest_interpolate(size).astype(np.uint8)

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
