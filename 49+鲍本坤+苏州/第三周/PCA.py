import matplotlib.pyplot as plt
import sklearn.decomposition as dp
import numpy as np
from sklearn.datasets import  load_iris
#本例子为直接调用PCA算法实例
x= np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca= dp.PCA(n_components=3)  #PCA降为2  #加载pca算法，设置降维后主成分数目为2
#相当于先调用fit再调用transform 对原始数据进行降维，保存在reduced_x中  ,对列压缩的
reduce_x = pca.fit_transform(x)
# m,n=[],[]
# for i in range(len(reduce_x)):
#     m.append(reduce_x[i][0])
#     n.append(reduce_x[i][1])
# plt.scatter(n,m,c='r')
# plt.show()
print(reduce_x)