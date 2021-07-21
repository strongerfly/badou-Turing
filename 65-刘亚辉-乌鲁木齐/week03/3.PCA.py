import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import sklearn.decomposition as dp


class PCA():
    def __init__(self, X, K):
        self.X = X                                      # 原始矩阵
        self.K = K                                      # 降为K维
        self.centralized = self._centralized()          # 中心化矩阵
        self.cov = self._coved()                        # 协方差矩阵
        self.eig_value, self.eig_vector = self._eig()   # 特征值及特征向量
        self.rotate = self._rotate()                    # 旋转矩阵
        self.dim_reduce = self._dim_reduce()            # 降维矩阵

    def _centralized(self):
        # step 1: 中心化矩阵
        mean = np.array([np.mean(col) for col in self.X.T])
        return self.X - mean

    def _coved(self):
        # step 2: 协方差矩阵
        cov = self.centralized.T.dot(self.centralized) / (self.centralized.shape[0] - 1)
        return cov

    def _eig(self):
        # step 3: 求特征值及特征向量
        eig_value, eig_vector = np.linalg.eig(self.cov)
        return eig_value, eig_vector

    def _rotate(self):
        # 求转换矩阵U
        # 对特征值从大到小排列
        index = np.argsort(-1*self.eig_value)
        # 对特征向量按特征值从大到小挑选特征列，得到降维转换矩阵
        UT = [self.eig_vector[:, index[i]] for i in range(self.K)]
        return np.transpose(UT)

    def _dim_reduce(self):
        # 转换后的矩阵
        return np.dot(self.X, self.rotate)


def iris_classify(reduced_x, y):
    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
    for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i]==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    return red_x, red_y, blue_x, blue_y, green_x, green_y


def iris_plot(red_x, red_y, blue_x, blue_y, green_x, green_y):
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')


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
# 使用X矩阵测试PCA算法
# pca = PCA(X, 2)

# 使用鸢尾花数据测试
# 使用自己实现的PCA算法
iris_x, iris_y = load_iris(return_X_y=True)
iris_pca = PCA(iris_x, 2)   # 对数据进行降维处理
# 根据标签分类降维后数据，查看降维效果
red_x, red_y, blue_x, blue_y, green_x, green_y = iris_classify(iris_pca.dim_reduce, iris_y)
plt.subplot(121).set_title("Iris Classify(by hand)")
iris_plot(red_x, red_y, blue_x, blue_y, green_x, green_y)

# 调用sklearn实现
sk_pca = dp.PCA(n_components=2)                 # 加载PCA算法，并设置降维后的主成分维度为2
sk_reduced_x = sk_pca.fit_transform(iris_x)     # 对数据进行降维处理
red_x, red_y, blue_x, blue_y, green_x, green_y = iris_classify(sk_reduced_x, iris_y)
plt.subplot(122).set_title("Iris Classify(SKLEARN)")
iris_plot(red_x, red_y, blue_x, blue_y, green_x, green_y)

plt.show()
