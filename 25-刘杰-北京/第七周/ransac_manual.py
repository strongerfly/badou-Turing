#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：ransac_manual.py
@Author  ：luigi
@Date    ：2021/8/2 下午4:02 
'''

import numpy as np
import matplotlib.pyplot as plt


class liner():
    """定义线性类"""

    # 通过最小二乘法拟合线性关系
    def fit(self, data):
        x = data[0]
        y = data[1]
        A = np.vstack([x, np.ones(len(x))]).T
        self.m, self.c = np.linalg.lstsq(A, y, rcond=None)[0]
        return self.m, self.c

    # 线性预估
    def predict(self, X):
        return X * self.m + self.c


def ransac(data, model, sample_number, epoch, threshold):
    """ ransac算法实现

    :param data: 数据集
    :type data: np.ndarray
    :param model: 模型
    :type model: class type
    :param sample_number: 随机采样样本数
    :type sample_number: int
    :param epoch: 迭代次数
    :type epoch: int
    :param threshold: 判断内群的阈值
    :type threshold: int
    :return: 模型
    :rtype: class type
    """

    max = 0
    target = None
    for i in range(epoch):
        # 根据参数sample_number，选择k个随机点作为内群
        dataIndex = np.arange(data.shape[0])
        dataIndexRandomk = np.random.choice(dataIndex, sample_number)
        dataRandomK = data[dataIndexRandomk]

        # 选取除k个随机点之外的所有点作为验证模型的数据点

        # 方式一：通过list generation
        # dataRandomExcept = data[[i for i in dataIndex if i not in dataIndexRandomk]]
        # 方式二：通过numpy mask
        mask = np.ones(data.shape[0], dtype=bool)
        mask[dataIndexRandomk] = False
        dataRandomExcept = data[mask]
        valX = dataRandomExcept[:, 0]
        valY = dataRandomExcept[:, 1]

        # 模型拟合
        model.fit(dataRandomK)
        # 模型预估
        predictY = model.predict(valX)
        # 损失函数
        cost = np.absolute(valY - predictY)
        # 计算内群数
        count = np.sum(cost <= threshold)

        if count > max:
            max = count
            print('max is：{}'.format(count))
            target = model

    return target


def main():
    model = liner()
    data = np.random.randint(1, 100, (100, 2))
    sample_k = 5
    epoch = 10000
    threshold = 5
    ransac(data, model, sample_k, epoch, threshold)

    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'o', label='original data', markersize=10)
    plt.plot(x, model.m * x + model.c, 'r', label='fitter line')
    plt.show()


if __name__ == '__main__':
    main()
