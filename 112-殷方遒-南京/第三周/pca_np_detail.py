# 第三周作业 pca实现 numpy公式实现
import numpy as np

'''
求矩阵X的K阶降维矩阵Z
'''


class CPCA(object):  # CPCA类 继承object类
    '''
    note:输入矩阵X shape(m,n) m:矩阵行数(不同样本)，n:矩阵列数(不同特征)
    '''

    def __init__(self, X, K):  # 初始化矩阵X,降维维度K
        '''
        :param X,样本矩阵
        :param K,矩阵X的降维维度
        '''
        self.X = X  # 样本矩阵
        self.K = K  # 降维维度
        self.centrX = []  # 样本中心化矩阵
        self.C = []  # 样本协方差矩阵
        self.U = []  # 样本转换矩阵
        self.Z = []  # 样本K阶降维矩阵

        self.centrX = self._centr()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centr(self):  # 中心
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 转置均值
        c = self.X - mean  # 矩阵各值与均值的差
        # c = self.X - self.X.mean(axis=0)
        return c

    def _cov(self):  # 协方差
        #n = np.shape(self.centrX)[0]  # 中心化矩阵样本数
        c = np.dot(self.centrX.T, self.centrX) / self.X.shape[0]  # 中心化样本矩阵协方差公式
        return c

    def _U(self):  # 转换矩阵
        eig_values, eig_vectors = np.linalg.eig(self.C)  # 调用eig()计算协方差的特征值,特征向量

        index = np.argsort(-eig_values)  # 特征值降序索引

        new_vecs = eig_vectors[:, index[:self.K]]  # 新特征空间

        # 另一种写法
        # new_vecs = [eig_vectors[:, index[i]] for i in range(self.K)]  # 新特征空间
        # u = np.transpose(new_vecs)  # 转置

        return new_vecs

    def _Z(self):  # 降维矩阵
        z = np.dot(self.X, self.U)
        return z


if __name__ == "__main__":
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    K = np.shape(X)[1] - 2
    pca = CPCA(X, K)
    print(pca.Z)  # 为啥降维结果不一样
