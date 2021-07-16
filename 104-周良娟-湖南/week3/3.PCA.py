import numpy as np
import math
import copy

def PCA(data, n_components):
    # 数据标准化 按列求均值
    data_mean = data.mean(axis = 0)
    h, w = data.shape
    data = data - data_mean
    data_covariance = np.dot(data.T, data) / h   # 协方差阵
    #计算特征值和特征向量
    eig_vals,eig_vectors = np.linalg.eig(data_covariance)
    # 对所得的特征值进行降序排列，
    ord_eig_vals = np.argsort(-eig_vals)
    # 获得排序后的特征向量
    data_components = eig_vectors[:, ord_eig_vals[:n_components]]
    print(data - data_mean)
    return eig_vals[ord_eig_vals], data_components, np.dot(data, data_components)

if __name__ == '__main__':
    data = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
    eig_vals, data_components, new_data = PCA(data, 2)
    print('排序后的特征值：\n', eig_vals)                  #输出降维后的数据
    print('排序后的特征值对应的2个特征向量：\n', data_components)
    print('新的降维后的矩阵:\n', new_data)







