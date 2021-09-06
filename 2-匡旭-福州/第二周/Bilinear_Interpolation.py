import cv2
import numpy as np
import matplotlib.pyplot as plt

def Bilinear_Inter(src_data, dst_height, dst_width) :
    ori_height, ori_width, channel = src_data.shape
    if dst_width == ori_width and dst_height == ori_height:
        return src_data.copy()

    ratio_height = ori_height / dst_height
    ratio_width  = ori_width /  dst_width
    dst_data = np.zeros((dst_height, dst_width, channel), dtype=np.uint8)
    for i in range(channel) :
        for y in range(dst_height) :
            for x in range(dst_width) :
                x_ori = (x + 0.5)* ratio_width - 0.5
                y_ori = (y + 0.5 ) * ratio_height - 0.5
                # 计算在原图上四个近邻点的位置
                x_ori_0 = int(np.floor(x_ori))
                y_ori_0 = int(np.floor(y_ori))
                x_ori_1 = min(x_ori_0 + 1, ori_width - 1)
                y_ori_1 = min(y_ori_0 + 1, ori_height - 1)

                # 双线性插值
                value0 = (x_ori_1 - x_ori) * src_data[y_ori_0, x_ori_0, i] + (x_ori - x_ori_0) * src_data[y_ori_0, x_ori_1, i]
                value1 = (x_ori_1 - x_ori) * src_data[y_ori_1, x_ori_0, i] + (x_ori - x_ori_0) * src_data[y_ori_1, x_ori_1, i]
                dst_data[y, x, i] = int((y_ori_1 - y_ori) * value0 + (y_ori - y_ori_0) * value1)
    return dst_data

def Bilinear_Inter_OpenCV(src_data, dst_height, dst_width) :
    ori_height, ori_width, channel = src_data.shape
    dst_data = cv2.resize(src_data, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
    return dst_data
if __name__ == '__main__' :
    path = 'lenna.png'
    height = 540
    width  = 960
    image = cv2.imread(path)
    bi_img1 = Bilinear_Inter(image, height, width)
    bi_img2 = Bilinear_Inter_OpenCV(image, height, width)
    cv2.imshow('src_image', image)
    cv2.imshow('dst_image', bi_img1)
    cv2.imshow('dst_image2', bi_img2)
    cv2.waitKey()
