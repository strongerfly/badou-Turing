# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 21:08:02 2021

@author: wp
"""

import numpy as np

def PCA(data, n_comps):
    data_mean = data.mean(axis=0) # 按列求均值
    h, w = data.shape
    data_c = data - data_mean
    data_covariance = np.dot(data_c.T, data_c) / h  # 协方差矩阵
    eig_vals, eig_vectors = np.linalg.eig(data_covariance)
    ord_eig_vals = np.argsort( - eig_vals)
    data_components = eig_vectors[:, ord_eig_vals[:n_comps]]
    return eig_vals[ord_eig_vals], data_components, np.dot(data_c, data_components)

if __name__ == '__main__':
    data = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
    eig_vals, data_components, new_data = PCA(data, 2)
    print('排序后的特征值：\n', eig_vals)                  #输出降维后的数据
    print('贡献率： \n',eig_vals[:2] / np.sum(eig_vals))
    print('排序后的特征值对应的2个特征向量：\n', data_components)
    print('降维后的新矩阵:\n', new_data)
