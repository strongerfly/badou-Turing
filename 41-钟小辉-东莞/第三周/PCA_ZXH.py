import numpy as np
import  pandas as pd
from sklearn.preprocessing import StandardScaler
import  matplotlib.pyplot as plt


class PCA_np():

    #确定降维目标
    def __init__(self,n_components):
        self.n_components =n_components

    #如何将输入data 降维，根据公式Ax = λx
    def fit_transform(self,X):
        #原特征维度
        self.n_feature = X.shape[1]

        # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
        X_std = StandardScaler().fit_transform(X)
        X = X_std - np.mean(X_std, axis=0)  # axis = 0 输出矩阵为1行，求的是每一列的均值；如果是axis=1，输出矩阵是1列，求每一行的均值
        # print(X_std)
        #求取协方差矩阵,使得X均值为0，方差为1.
        # X = X-np.mean(X, axis=0)  # axis = 0 输出矩阵为1行，求的是每一列的均值；如果是axis=1，输出矩阵是1列，求每一行的均值
        self.cov_mat = X.T.dot(X) /X.shape[0]
        # print(self.cov_mat)
        #求得特征向量及值
        eig_vals,eig_vecs = np.linalg.eig(self.cov_mat)
        # print("values:  \n%s" %eig_vals)
        # print("vectors: \n%s" %eig_vecs)

        #取特征向量的前n_components维度
        idx = np.argsort(-eig_vals)
        # print(idx)
        self._components = eig_vecs[:,idx[:self.n_components]]
        # print(self._components)

        return X.dot(self._components)


#1.学习numpy如何实现PCA
data =  pd.read_csv("iris.data")
X = data.iloc[:,0:4].values
y = data.iloc[:,4].values
print(X.shape)
pca = PCA_np(n_components =2)
new_x = pca.fit_transform(X)
# print(new_x)

#原始数据
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()

#结果数据
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(new_x[y==lab, 0],
                new_x[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
