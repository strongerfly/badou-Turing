# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 22:20:03 2021

@author: wp
"""

import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit_transform(self, X):
        X = X - X.mean(axis = 0)
        covariance = np.dot(X.T, X) / X.shape[0]
        eig_vals, eig_vectors = np.linalg.eig(covariance)
        idx = np.argsort(-eig_vals)
        components = eig_vectors[:, idx[:self.n_components]]
        
        return idx, components, np.dot(X, components)
    
pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
idx, new_copms, new_X = pca.fit_transform(X)