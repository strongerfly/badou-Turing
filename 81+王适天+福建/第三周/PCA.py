#    @author Created by Genius_Tian

#    @Date 2021/7/5

#    @Description PCA主成分分析——降维算法
from typing import Optional, Any
from sklearn import datasets
import cv2
import numpy as np


class PCA:
    components_: Optional[Any]
    covariance: Optional[Any]
    n_features: int

    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features = X.shape[1]
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / (X.shape[0] - 1)
        eig_values, eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(eig_values)
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        return np.dot(X, self.components_)


if __name__ == '__main__':
    iris = datasets.load_iris()
    pca = PCA(2)
    transform = pca.fit_transform(iris.data)
    print(transform)
