import cv2
import numpy as np
import matplotlib.pyplot as plt


# python实现的最近邻插值法
def Nearest_Neighbor_Inter(src_data, dst_height, dst_width):
    ori_height, ori_width, channel = src_data.shape
    ratio_height = ori_height / dst_height
    ratio_width = ori_width / dst_width
    dst_data = np.zeros((dst_height, dst_width, channel), np.uint8)
    for i in range(channel):
        for y in range(dst_height):
            for x in range(dst_width):
                x_ori = int(x * ratio_width)
                y_ori = int(y * ratio_height)
                dst_data[y, x, i] = src_data[y_ori, x_ori, i]

    return dst_data

def Nearest_Neighbor_Inter_Class(src_data, dst_height, dst_width):
    ori_height, ori_width, channels =  src_data.shape
    dst_data = np.zeros((dst_height, dst_width, channels), np.uint8)
    ratio_height = ori_height / dst_height
    ratio_width = ori_width / dst_width
    for i in range(dst_height) :
        for j in range(dst_width) :
            x = int(i * ratio_width)
            y = int(j * ratio_height)
            dst_data[j, i] = src_data[y, x]

    return dst_data

# OpenCV实现最近邻插值
def Nearest_Neighbor_Inter_OpenCV(src_data, dst_height, dst_width):
    dst_data = cv2.resize(src_data, (dst_width, dst_height), interpolation=cv2.INTER_NEAREST)
    return dst_data

if __name__ == '__main__' :
    path = 'lenna.png'
    height = 128
    width = 128
    # return [height, width, channel]的numpy.ndarray对象, height表示图片高度，width表示图片宽度，channel表示图片通道
    img = cv2.imread(path)
    img1 = Nearest_Neighbor_Inter(img, height, width)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # matplotlib 实现图片显示
    plt.imshow(img1)
    plt.show()


    # OpenCV实现图片显示
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = Nearest_Neighbor_Inter_OpenCV(img, height, width)
    img3 = Nearest_Neighbor_Inter_Class(img, height, width)
    cv2.imshow('result1', img1)
    cv2.imshow('result2', img2)
    cv2.imshow('result3', img3)
    cv2.waitKey(0)
