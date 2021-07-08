# -*- coding: utf-8 -*-

from PIL import Image

import numpy as np

"""
    函数功能：将原图片变为灰度图片
    参数说明：
        path：原图片路径
    输出说明：
        data：灰度图片像素数组
"""
def loadImage(path):
    # 打开图片
    img = Image.open(path)
    # 将图像转换成灰度图
    img = img.convert("L")
    # 图像对象，可变为数组
    data = img.getdata()
    # size中1为长，0为宽
    # 为了避免溢出，这里对数据进行一个缩放，缩小100倍
    data = np.array(data).reshape(img.size[1], img.size[0])/100
    # 查看原图的话，需要还原数据
    new_im = Image.fromarray(data*100)
    # 将图片保存在路径中
    # 一定要写convert()不然会一直报错cannot write mode F as JPEG或者keyerror
    new_im.convert('RGB').save('pca_gray.jpg')
    # 展示灰度图片
    # new_im.show()
    return data

"""
    函数功能：使用pca对灰度图片进行降维处理
    参数说明：
        data：灰度图路径
        k：主成分个数
    输出说明：
        pass
"""
def pca(data, k):
    # 求图片每一列的均值
    mean = np.array([np.mean(data[:, index]) for index in range(data.shape[1])])
    # 去中心化
    normal_data = data - mean
    # 得到协方差矩阵：1/n＊(X * X^T)，这里不除以n也不影响
    matrix = np.dot(np.transpose(normal_data), normal_data)/normal_data.shape[0]
    eig_val, eig_vec = np.linalg.eig(matrix)
    eig_index = np.argsort(eig_val)
    # 取下标的倒数k位，也就是取前k个大特征值的下标
    eig_vec_index = eig_index[:-(k+1):-1]
    # 取前k个大特征值的特征向量
    feature = eig_vec[:, eig_vec_index]
    # 将特征值与对应特征向量矩阵乘得到最后的pca降维图
    new_data = np.dot(normal_data, feature)
    # 将降维后的数据映射回原空间
    rec_data = np.dot(new_data, np.transpose(feature)) + mean
    # 压缩后的数据也需要乘100还原成RGB值的范围
    newImage = Image.fromarray(np.uint8(rec_data*100))
    # 将处理好的降维图片存入文件夹
    newImage.convert('RGB').save( 'E:/badou-Turing/38-黄恸-广州/week3/1/'+ 'k=' + str(k) + '.jpg')
    

if __name__ == '__main__':
    data = loadImage('E:/lenna.png')

    for i in range(1, data.shape[1]+1):
        pca(data, i)
        print('正在处理第', str(i) + '/' + str(data.shape[1]), '张图片')

    print('处理完成，请查看pcaImg文件夹')
