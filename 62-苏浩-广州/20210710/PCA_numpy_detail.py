"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""
 
import numpy as np


class CPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''
    def __init__(self, X, K):
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        # 样本矩阵X
        self.X = X
        # K阶降维矩阵的K值
        self.K = K
        # 矩阵X的中心化
        self.centrX = []
        # 样本集的协方差矩阵C
        self.C = []
        # 样本矩阵X的降维转换矩阵
        self.U = []
        # 样本矩阵X的降维矩阵Z
        self.Z = []
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        # Z=XU求得
        self.Z = self._Z()
        
    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        # 样本集的特征均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:\n',mean)
        # 样本集的中心化
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX
        
    def _cov(self):
        '''
        求样本矩阵X的协方差矩阵C
        '''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C
        
    def _U(self):
        '''
        求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        '''
        # 先求X的协方差矩阵C的特征值和特征向量
        # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/gene
        a,b = np.linalg.eig(self.C)
        rated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        #给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U
        
    def _Z(self):
        '''
        按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
        '''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
        
if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)
