# coding: utf-8
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
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
     [0.1956, 0.4280]
     ]

# # 输出数据集
# print (X)



plt.rcParams['font.size'] = 14
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

x = [n[0]for n in X]
y = [n[1]for n in X]
print(x)
print(y)

#显示图像
titles = [u'聚类图像 K=2', u'聚类图像 K=3',
          u'聚类图像 K=4', u'聚类图像 K=5',u'聚类图像 K=6', u'聚类图像 K=7']

for k in range(6):

     km = KMeans(n_clusters=k+2).fit(X)
     # print(f"labels_ = {km.labels_}")
     _labels = km.labels_
     cluster_centers = km.cluster_centers_
     # print(f"cluster_centers = {cluster_centers}")
     plt.subplot(2,3,k+1)
     plt.scatter(x,y,c=_labels,marker="o")
     plt.scatter([n[0]for n in cluster_centers],[n[1]for n in cluster_centers],  linewidths=3, marker='+', s=300, c='black')
     plt.title = titles[k]
     plt.xticks([]), plt.yticks([])

plt.show()

#评估,score数值越大越好
scores = []
for k in range(2,10):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

print(scores)
