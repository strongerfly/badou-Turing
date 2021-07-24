# -*- coding:utf-8 -*-
"""
自实现k-means时，中心点的算法一定要好，不然会稀烂
中心点的选择方式： https://blog.csdn.net/zhihaoma/article/details/48649489
"""
import math
import sys

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


X = [[0.0888, 0.5885],
     # [0.1399, 0.8291],
     # [0.1329, 0.8491],
     # [0.1299, 0.7991],
     # [0.1369, 0.8221],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]]
X = np.array(X)


def k_means_by_sklearn(n=3):
    km = KMeans(n_clusters=n)
    y_pred = km.fit_predict(X)
    print("y_pred: {}".format(y_pred))
    # plt.scatter(x=X[:, 0], y=X[:, 1], c=y_pred, marker="*")
    x, y = X[:, 0], X[:, 1]
    for i in range(n):
        plt.scatter(x=x[y_pred == i], y=y[y_pred == i], marker="*")
    plt.legend(['A', 'B', 'C'])
    plt.show()


class KMeansDiy:
    def __init__(self, n_cluster=3):
        self.n_cluster = n_cluster
        self.last_center = []
        # 选中的中心点  {0: (x, y)}
        self.centers = {}
        # 每个中心里所拥有的点下标
        self.points = {}

    def fit_predict(self, x, y):
        while True:
            eq = self.calc_center(x, y)
            if eq:
                res = np.zeros(len(x))
                for c, v in self.points.items():
                    res[v] = c
                return res
            else:
                self.points = {i: [] for i in range(self.n_cluster)}
                self.calc_points(x, y)

    def calc_points(self, x, y):
        for i in range(len(x)):
            p = [x[i], y[i]]
            nearest, index = sys.maxsize, None
            for _index, centers in self.centers.items():
                l2 = math.sqrt((centers[0] - p[0])**2 + (centers[1] - p[1])**2)
                if l2 < nearest:
                    nearest, index = l2, _index
            self.points[index].append(i)
        print(self.points)

    def calc_center(self, x, y):
        if not self.points:
            self.centers = {i: [x[i], y[i]] for i in range(self.n_cluster)}
        else:
            self.last_center = list(self.centers.values())
            self.last_center.sort()
            for k, indexes in self.points.items():
                d = [[x[i], y[i]] for i in indexes]
                self.centers[k] = np.average(np.array(d), 0).tolist()
        print("centers: {}".format(self.centers))
        curr_center = list(self.centers.values())
        curr_center.sort()
        return self.last_center == curr_center


def test(n=3):
    km = KMeansDiy(n_cluster=n)
    x, y = X[:, 0], X[:, 1]
    y_pred = km.fit_predict(x, y)
    print("y_pred: {}".format(y_pred))
    for i in range(n):
        plt.scatter(x=x[y_pred == i], y=y[y_pred == i], marker="*")
    plt.legend(['A', 'B', 'C'])
    plt.show()


if __name__ == '__main__':
    # k_means_by_sklearn()
    test()
