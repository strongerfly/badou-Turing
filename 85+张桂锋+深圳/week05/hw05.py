import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
'''
第五周作业：
1）KMeans实现
'''
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
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
print("y_pred = ",y_pred)
x = [n[0] for n in X]
print (x)
y = [n[1] for n in X]
print (y)

# 坐标
x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

# 分布获取类标为0、1、2的数据 赋值给(x1,y1) (x2,y2) (x3,y3)
i = 0
while i < len(X):
     if y_pred[i] == 0:
          x1.append(X[i][0])
          y1.append(X[i][1])
     elif y_pred[i] == 1:
          x2.append(X[i][0])
          y2.append(X[i][1])
     elif y_pred[i] == 2:
          x3.append(X[i][0])
          y3.append(X[i][1])
     i = i + 1

plot1, = plt.plot(x1, y1, 'or')
plot2, = plt.plot(x2, y2, 'og')
plot3, = plt.plot(x3, y3, 'ob')

#plt.scatter(x, y, c=y_pred, marker='o')
plt.title("Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend((plot1, plot2, plot3), ('A', 'B', 'C'), fontsize=10)
plt.show()


