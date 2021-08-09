import numpy as np
import sklearn.decomposition as dp

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 获取特征(属性)的数量
        self.n_features = X.shape[1]
        # 1-对原始数据零均值化(中心化)
        X = X - X.mean(axis=0)
        # 2-求协方差矩阵
        self.covariance = np.dot(X.T, X) / X.shape[0]
        # 3-对协方差矩阵求特征值 eig_vals 和特征向量eig_vectors
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 4-获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 5-降维矩阵
        self.componemts_ = eig_vectors[:, idx[:self.n_components]]
        # 6-对 X 进行降维并返回
        return np.dot(X, self.componemts_)

if __name__ == '__main__':

    pca = PCA(n_components=2)
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    newX = pca.fit_transform(X)
    print(newX)


    """
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    pca = dp.PCA(n_components=2)
    pca.fit(X)
    newX = pca.fit_transform(X)
    print("输出贡献率：", pca.explained_variance_ratio_)
    print("降维后的矩阵：\n", newX)
    """