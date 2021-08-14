# 第三周作业 pca实现 numpy

import numpy as np


class PCA():  # PCA类
    def __init__(self, n_components):  # 初始化
        self.n_components = n_components  # [成分数]

    def fit_transform(self, X):  # 拟合变换矩阵X
        # self.n_featrues=X.shape[1]#[特征数](shape[1]:矩阵列数 shape[0]:矩阵行数)
        X = X - X.mean(axis=0)  # 各列值与其均值的差 每行为一个样本 每列为一个维度
        self.covariance = np.dot(X.T, X) / X.shape[0]  # [协方差矩阵] 是不同维度之间的协方差
        eig_values, eig_vectors = np.linalg.eig(self.covariance)  # eig()方法:返回特征值,特征向量
        index = np.argsort(-eig_values)  # 降序索引
        self.components_ = eig_vectors[:, index[:self.n_components]]  # 特征向量提取
        return np.dot(X, self.components_)  # 降维X


pca = PCA(n_components=2)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
dimX = pca.fit_transform(X)
print(dimX)  # 不知道哪里有问题 和其他pca输出结果第二维数据符号相反
