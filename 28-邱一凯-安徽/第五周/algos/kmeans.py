#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Project ：badou-Turing 
@File    ：kmeans.py
@Author  ：Autumn
@Date    ：2021/7/21 19:32 
"""
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)

marks = ['o', '*', '+', 's', 'x', 'd']
colors = ["red", "blue", "cyan", "green", "pink", "black"]

iris_data = datasets.load_iris().data

fig = plt.figure()
fig.suptitle("K-means 聚类", fontproperties=font_set)
# 绘制数据分布图
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_title("原图", fontproperties=font_set)
ax1.scatter(iris_data[:, 0], iris_data[:, 1], c="blue", marker='o')

# k = 3 ,data =all_data
k_num = 3
k_means = KMeans(n_clusters=k_num).fit(iris_data)
ax2 = fig.add_subplot(3, 2, 2)
ax2.set_title("3 ,all_data图", fontproperties=font_set)
label_pred = k_means.labels_
for i in range(k_num):
    ax2.scatter(iris_data[label_pred == i][:, 0], iris_data[label_pred == i][:, 1],
                c=colors[i], marker=marks[i])

# k = 4,data =all_data
k_num = 4
k_means = KMeans(n_clusters=k_num).fit(iris_data)
ax3 = fig.add_subplot(3, 2, 3)
ax3.set_title("4 ,all_data图", fontproperties=font_set)
label_pred = k_means.labels_
for i in range(k_num):
    ax3.scatter(iris_data[label_pred == i][:, 0], iris_data[label_pred == i][:, 1],
                c=colors[i], marker=marks[i])

# k = 3,data = [:,0:2]
k_num = 3
k_means = KMeans(n_clusters=k_num).fit(iris_data[:, 0:2])
ax4 = fig.add_subplot(3, 2, 4)
ax4.set_title("3 ,data[:,0:2]图", fontproperties=font_set)
label_pred = k_means.labels_
for i in range(k_num):
    ax4.scatter(iris_data[label_pred == i][:, 0], iris_data[label_pred == i][:, 1],
                c=colors[i], marker=marks[i])

# k = 3,data = [:,3:]
k_num = 3
k_means = KMeans(n_clusters=k_num).fit(iris_data[:, 3:])
ax5 = fig.add_subplot(3, 2, 5)
ax5.set_title("3 ,data[:,3:]图", fontproperties=font_set)
label_pred = k_means.labels_
for i in range(k_num):
    ax5.scatter(iris_data[label_pred == i][:, 0], iris_data[label_pred == i][:, 1],
                c=colors[i], marker=marks[i])

# k = 4,data = [:,0:2]
k_num = 4
k_means = KMeans(n_clusters=k_num).fit(iris_data[:, 3:])
ax6 = fig.add_subplot(3, 2, 6)
ax6.set_title("4 ,data[:,0:2]图", fontproperties=font_set)
label_pred = k_means.labels_
for i in range(k_num):
    ax6.scatter(iris_data[label_pred == i][:, 0], iris_data[label_pred == i][:, 1],
                c=colors[i], marker=marks[i], label="label" + str(i))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)
plt.legend(loc=2)
plt.savefig("../pict/k-means.png")
plt.show()
