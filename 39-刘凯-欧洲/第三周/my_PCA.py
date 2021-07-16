# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

class my_PCA():
    def __init__(self, n_components):
        # number of the features after PCA
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        self.n_samples = X.shape[0]
        # 数据中心化，按列来求取均值。就是不用样本之间，同一特征的均值
        X = X - X.mean(axis=0)
        # 求协方差矩阵
        self.covariance = np.dot(X.T, X) / (self.n_samples - 1)
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        # 对X进行降维
        return np.dot(X, self.components_)


# 调用
pca_numpy = my_PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX_numpy = pca_numpy.fit_transform(X)

# PCA with sklearn
pca_sklearn = PCA(n_components=2)   #降到2维
pca_sklearn.fit(X)                  #训练
newX_sklearn = pca_sklearn.fit_transform(X)   #降维后的数据


print(pca_sklearn.explained_variance_ratio_)  #输出贡献率
print(newX_numpy)                  #输出降维后的数据
print(newX_sklearn)