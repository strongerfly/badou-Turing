#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris
import numpy as np

def iris_demo():
    x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
    print('样本数量：',x.shape,'标签：',y.shape)
    pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
    reduced_x=pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中
    print(reduced_x[:4, :])
    print('输出贡献率:',pca.explained_variance_ratio_)  # 输出贡献率,什么含义？

    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
    for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i]==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    plt.show()

class PCA_NP():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1] #特征空间维度
        self.m = X.shape[0] #样本数量
        # 求协方差矩阵
        X = X - X.mean(axis=0) #中心化
        self.covariance = np.dot(X.T, X) / self.m  # 0均值化后的协方差矩阵公式
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print(eig_vals)
        print(eig_vectors)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals) #基本是对称正定矩阵，特征值一般为正数？
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print(self.components_)
        # 对X进行降维
        #print(X.shape,self.components_.shape)
        return np.dot(X, self.components_)

def my_iris_demo():
    x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
    pca =PCA_NP(n_components=2)
    reduced_x = pca.fit_transform(x)
    print(reduced_x[:4,:])#第二维坐标跟iris_demo是正负反过来的？

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

if __name__ == "__main__":
    #iris_demo()
    my_iris_demo()