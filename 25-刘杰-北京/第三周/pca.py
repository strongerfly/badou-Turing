#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：pca.py
@Author  ：luigi
@Date    ：2021/7/7 上午11:17 
'''

import numpy as np
class PCA():
    """ PCA算法实现
    """

    def __init__(self,X,k):
        """初始化PCA类，并执行1.中心化，2.构造协方差矩阵，3.提取特征值与特征向量，4，降维

        :param X: 输入的特征矩阵，行表示样本，列表示特征
        :type X: np.ndarray
        :param k: 降维的目标维度
        :type k: int
        """
        self.X = X
        assert self.X.ndim == 2
        self.k = k
        self.X_shape = self.X.shape
        self.X_samples = self.X_shape[0]
        self.X_features = self.X_shape[1]

        self.centX = self._centralize()
        self.convM = self._cov()
        self.convF_value,self.convF_vector = self._extract()
        self.Z_features = self._reduced()

    def _centralize(self):
        """中心化
        :return: 中心化的矩阵，即原始矩阵没个元素按列减去该列的均值
        :rtype: np.ndarray
        """
        X_mean = np.mean(self.X,axis=0)
        X_centralized = self.X - X_mean
        print("centralized X:\n{}".format(X_centralized))
        return X_centralized

    def _cov(self):
        """构造协方差矩阵

        :return: 协方差矩阵
        :rtype: np.ndarray
        """
        # matrix = np.zeros((features.shape[-1],features.shape[-1]))
        # for i in np.arange(features.shape[-1]):
        #     for j in np.arange(features.shape[-1]):
        #         matrix[i,j]=np.sum(features[:,i]*features[:,j])/(features.shape[0]-1)
        con_matrix = np.dot(self.centX.transpose(), self.centX)/(self.X_samples-1)
        print("covariance matrix:\n{}".format(con_matrix))
        return con_matrix

    def _extract(self):
        """ 特征值与特征向量为矩阵的两个维度的特征，本方法用于提取这两个特征

        :return: 特征值与特征向量
        :rtype: tuple
        """
        f_value, f_vectors = np.linalg.eig(self.convM)
        print("feature values of covariance matrix:\n{}".format(f_value))
        print("feature vectors of covariance matrix:\n{}".format(f_vectors))
        return f_value, f_vectors

    def _reduced(self):
        """ 降维，只取排名靠前的部分特征向量，与原特征矩阵的内积，得到新的特征矩阵

        :return: 降维后的特征
        :rtype: np.ndarray
        """
        #保留的主成分所对应的的降序索引
        main_index = self.convF_value.argsort()[::-1][:self.k]
        target_features = np.dot(self.X,self.convF_vector[main_index].T)
        print("target feature matrix:\n{}".format(target_features))
        return target_features



features =np.arange(200).reshape((40,5))
target_feature_num = features.shape[1]-1
pca = PCA(features,target_feature_num)




