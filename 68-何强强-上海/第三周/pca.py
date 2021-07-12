# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class PCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, x, k):
        '''
        :param x,训练样本矩阵X
        :param k,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.x = x  # 样本矩阵X
        self.k = k  # K阶降维矩阵的K值
        self.center_x = []  # 矩阵X的中心化
        self.c = []  # 样本集的协方差矩阵C
        self.u = []  # 样本矩阵X的降维转换矩阵
        self.z = []  # 样本矩阵X的降维矩阵Z

        self.center_x = self._centralized()
        self.c = self._cov()
        self.u = self._u()
        self.z = self._z()  # z=XU求得

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.x)
        # 样本集的特征均值  x.T 矩阵的转置
        mean = np.array([np.mean(attr) for attr in self.x.T])
        print('样本集的特征均值:\n', mean)
        center_x = self.x - mean  # 样本集的中心化
        print('样本矩阵X的中心化center_x:\n', center_x)
        return center_x

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.center_x)[0]
        # 样本矩阵的协方差矩阵C   中心化的协方差矩阵可简化为 (z^T * z)/(n - 1)
        c = np.dot(self.center_x.T, self.center_x) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', c)
        return c

    def _u(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        a, b = np.linalg.eig(self.c)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        ut = [b[:, ind[i]] for i in range(self.k)]
        # transpose矩阵转置，与 ut.T效果一样
        u = np.transpose(ut)
        print('%d阶降维转换矩阵U:\n' % self.k, u)
        return u

    def _z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        z = np.dot(self.x, self.u)
        print('x shape:', np.shape(self.x))
        print('u shape:', np.shape(self.u))
        print('z shape:', np.shape(z))
        print('样本矩阵X的降维矩阵Z:\n', z)
        return z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    data = np.array([[10, 15, 29],
                     [15, 46, 13],
                     [23, 21, 30],
                     [11, 9, 35],
                     [42, 45, 11],
                     [9, 48, 5],
                     [11, 21, 14],
                     [8, 5, 15],
                     [11, 12, 21],
                     [21, 20, 25]])
    k = np.shape(data)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', data)
    pca = PCA(data, k)
