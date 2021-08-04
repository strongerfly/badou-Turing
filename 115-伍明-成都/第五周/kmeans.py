# -*- coding: utf-8 -*-
"""

@author: wuming
"""
#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
import matplotlib.pyplot as plt

def randCent(dataSet,k):
    #dataSet为数据集，k为聚类数
    n=dataSet.shape[1]
    centroids=np.zeros((k,n))
    # for i in range(k):
    #     rangi = np.random.randint(dataSet.shape[0])
    #     centroids[i,:] = dataSet[rangi,:]
    for i in range(n):
        min1=min(dataSet[:,i])
        rangi=float(max(dataSet[:,i]-min1))
        centroids[:,i]=(min1+rangi*np.random.rand(k,1)).reshape(k,)

    return centroids
def kMeans(dataSet,k):
    m=dataSet.shape[0]
    clusterAssment=np.zeros((m,2))
    centroids=randCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        # clusterChanged=False
        temp=centroids
        for i in range(m):
            minDist=np.inf
            minIndex=-1
            for j in range(k):
                dist=np.sqrt(sum(np.power(centroids[j,:]-dataSet[i,:],2)))
                if dist<minDist:
                    minDist=dist
                    minIndex=j
                # if clusterAssment[i,0]!=minIndex:
                #     clusterChanged=True
                clusterAssment[i,:]=int(minIndex),minDist**2

        for cent in range(k):
            ptsInClust=dataSet[clusterAssment[:,0]==cent]
            centroids[cent,:]=np.mean(ptsInClust,axis=0)
        if np.sum((temp-centroids)**2)<2:    #temp.all()==centroids.all():#kmeans终止条件
            clusterChanged=False
    return centroids,clusterAssment

if __name__=='__main__':
    X = np.random.randint(45, 70, (25, 2))
    Y = np.random.randint(60, 85, (25, 2))
    Z = np.vstack((X, Y))
    # convert to np.float32
    Z = np.float32(Z)
    k=4
    labels,clus=kMeans(Z,k)
    # print(clus)
    # print(len(clus))
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    fig, ax = plt.subplots()
    # 不同的子集使用不同的颜色
    for i in range(k):
        points = np.array([Z[j] for j in range(len(clus)) if clus[j,0] == i])
        ax.scatter(points[:, 0], points[:, 1], s=70, c=colors[i])
    ax.scatter(labels[:, 0], labels[:, 1], marker='*', s=200, c='black')
    plt.show()

