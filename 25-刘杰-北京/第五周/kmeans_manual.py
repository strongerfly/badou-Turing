#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：kmeans_manual.py
@Author  ：luigi
@Date    ：2021/7/22 下午5:02 
'''

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def _centralized(data, centers):
    """ kmeans算法的step#3，计算每个点到质心点的距离，最短距离重新分组为新的簇

    :param data: 数据集
    :type data: np.ndarray
    :param centers: 质心点
    :type centers: np.ndarray
    :return: 按照距质心点的最短距离重新分组
    :rtype: list
    """

    #计算数据集中的每个点到质心点的距离
    distances = np.array([np.hypot((data - center)[:, 0], (data - center)[:, 1]) for center in centers])
    cluster = []
    #按照最短距离从新分组
    for (i, A) in enumerate(distances):
        tmp = np.full(A.shape, True)
        for (j, B) in enumerate(distances):
            if (i != j):
                tmp = tmp & (A < B)
        cluster.append(data[tmp])

    return cluster


def kmeans(k, data):
    """

    :param k: 质心点的数量
    :type k: int
    :param data: 数据集
    :type data: np.ndarray
    :return: 最终簇和质心点
    :rtype: tuple
    """

    dataIndex = np.arange(data.shape[0])
    kCentersIndex = np.random.choice(dataIndex, k, replace=False)
    kCenters = data[kCentersIndex]
    print("init k centers: {}".format(kCenters))

    while True:
        kCluster = _centralized(data,kCenters)
        kCentersNew = np.array([np.mean(each,axis=0) for each in kCluster])
        print("new k centers: {}".format(kCentersNew))
        if(np.array_equal(kCenters, kCentersNew)):
            break
        kCenters = kCentersNew

    return kCluster,kCenters

def main():
    k = 4
    data,_ = make_blobs(n_samples=100, centers=k, cluster_std=0.60, random_state=0)
    # plt.scatter(data[:, 0], data[:, 1], s=50, cmap='viridis')
    print("your data is: {}".format(data))
    cluster,center = kmeans(k, data)

    for i in range(k):
        plt.scatter(cluster[i][:, 0], cluster[i][:, 1], c='blue', s=50)
    plt.scatter(center[:,0],center[:,1],c='red',s=200,alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()


