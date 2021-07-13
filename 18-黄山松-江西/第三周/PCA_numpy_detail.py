import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        self.X = X   # 样本矩阵X
        self.K = K   # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []    # 样本集的协方差矩阵C
        self.U = []    # 样本矩阵X的降维转换矩阵
        self.Z = []    # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()


    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]     # 获取数组的行数，即样本个数
        C = np.dot(self.centrX.T, self.centrX)/(ns-1)  # 样本矩阵的协方差矩阵C
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)  # 特征值赋给a，对应的特征向量赋给b。eig意思是矩阵特征值和特征向量。
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本矩阵的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1*a)  # arg意思是自变量。np.argsort()是返回数组值从小到大排序的索引值，此索引值是和排序之前是一样的。而括号里乘以-1就是从大到小排序。
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1    # 获取数组的列数，即维度数，然后减1
    print('样本集（10行3列，10个样例，每个样例3个特征）:\n', X)
    pca = CPCA(X, K)
