# 第三周作业 pca实现:sklearn.decomposition fit_transform()方法

import sklearn.decomposition as dp
import numpy as np

# 4维数据
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
# pca降维
pca = dp.PCA(n_components=2)  # 设置维度为2
reduce_x = pca.fit_transform(X)  # 降维数据
print(reduce_x)

# from sklearn.datasets._base import load_iris
# import matplotlib.pyplot as plt
# x, y = load_iris(return_X_y=True)  # 原数据,标签
# pca = dp.PCA(n_components=2)  # 设置维度为2
# reduce_x = pca.fit_transform(x)  # 降维数据
# red_x, red_y = [], []  # 红标签x,y维度空列表
# blue_x, blue_y = [], []  # 蓝标签x,y维度空列表
# green_x, green_y = [], []  # 绿标签x,y维度空列表
# for i in range(len(reduce_x)):  # 遍历样本
#     if y[i] == 0:  # 标签0
#         red_x.append(reduce_x[i][0])  # 降维数据值加到红标签x维度列表
#         red_y.append(reduce_x[i][1])  # 降维数据值加到红标签y维度列表
#     elif y[i] == 1:  # 标签1
#         blue_x.append(reduce_x[i][0])  # 降维数据值加到蓝标签x维度列表
#         blue_y.append(reduce_x[i][1])  # 降维数据值加到红标签y维度列表
#     else:  # 其他标签
#         green_x.append(reduce_x[i][0])  # 降维数据值加到绿标签x维度列表
#         green_y.append(reduce_x[i][1])  # 降维数据值加到绿标签y维度列表
# plt.scatter(red_x, red_y, c='r', marker='x')  # 设置红标签离散点(x,y,颜色,标识)
# plt.scatter(blue_x, blue_y, c='b', marker='D')  # 蓝标签离散点(x,y,颜色,标识)
# plt.scatter(green_x, green_y, c='g', marker='.')  # 绿标签离散点(x,y,颜色,标识)
# plt.show()  # 显示打开的图形
