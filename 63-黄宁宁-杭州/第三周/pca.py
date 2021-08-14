
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
'''
#sklearn
pca =PCA(n_components=1)
pca.fit(X) #训练
newX=pca.fit_transform(X) #降维后的数据
print(pca.explained_variance_ratio_)#输出贡献率
print(newX)
'''
#numpy
class CPCA(object):

    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features=X.shape[1]
        #求协方差矩阵
        X=X-np.mean(X,axis=0)
        self.cov=np.dot(X.T,X)/X.shape[0]
        #求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors=np.linalg.eig(self.cov)
        #对特征值进行降序排序
        id=np.argsort(-eig_vals)
        # 降维矩阵
        self.n_components_=eig_vectors[:,id[:self.n_components]]
        newX=np.dot(X,self.n_components_)
        return newX

pca=CPCA(n_components=2)
newX=pca.fit_transform(X)
print(newX)