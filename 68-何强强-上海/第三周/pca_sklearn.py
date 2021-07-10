# -*- coding:utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


if __name__ == '__main__':
    d = load_iris()
    pca = PCA(n_components=2)
    pca.fit(d.data)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    nd = pca.fit_transform(d.data)
    print(nd)
