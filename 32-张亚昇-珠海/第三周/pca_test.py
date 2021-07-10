import numpy as np
import math
from sklearn import datasets
import pandas as pd
#iris = datasets.load_iris()
#print(iris.data.shape)
#print(type(iris.data))
# iris_data = pd.DataFrame(iris.data)
#iris_data = iris.data
#X = iris_data
# print(iris_data.mean())
# print(iris_data.mean())
# print(iris_data)
# aaa = iris_data[:,2] - np.mean(iris_data[:,2])
#中心化
def center_standard(data):
    new_data = np.zeros(data.shape)
    for i in range(data.shape[1]):
        new_data[:, i] = data[:, i] - data[:, i].mean()
        # print(new_data)
        # np.append(total_data, new_data)

    return new_data

#求协方差矩阵，特征值，特征向量
def covfun(data, n_components):
    D = np.dot(data.T, data)
    cov = D / dd.shape[0]
    iris_val, iris_vector = np.linalg.eig(cov)
    new_iris_val = np.argsort(-iris_val)
    new_iris_vec = iris_vector[:, new_iris_val[:n_components]]
    return new_iris_vec

iris = datasets.load_iris()
iris_data = iris.data
dd = center_standard(iris_data)
vec = covfun(dd,2)
pca = np.dot(iris_data, vec)  #Xnew*W
print(pca)

# print(aaa.mean())
