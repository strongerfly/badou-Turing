import numpy as np


class PCA:
    def __init__(self, features):
        self.features = features

    def fit_transform(self, train):
        # 对原始数据零均值化
        train = train - train.mean(axis=0)
        # 求协方差矩阵
        cov = np.dot(train.T, train) / train.shape[0]
        # 求协方差矩阵的特征值和特征向量
        values, vectors = np.linalg.eig(cov)
        print(values)
        # 获取倒序排列的特征值序号
        ids_desc = np.argsort(values)
        features = vectors[:, ids_desc[: self.features]]
        return np.dot(train, features)


Pca = PCA(features=2)
train_set = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
train_pca = Pca.fit_transform(train_set)
print(train_pca)
