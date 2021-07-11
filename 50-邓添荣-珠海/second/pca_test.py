import numpy as np
from sklearn.decomposition import PCA


class PCA_class():
    def __init__(self, wei_number):
        self.wei_number = wei_number

    def fit_transform(self, X):
        #求得有几个维度
        self.weidu = X.shape[1]
        #求每个维度均值
        X = X - X.mean(axis=0)
        #求协方差矩阵
        self.xiefan_cha = np.dot(X.T, X)/X.shape[0]
        #求特征值和特征向量
        tezheng_humber, tezheng_fuction = np.linalg.eig(self.xiefan_cha)
        #对特征值排序
        idx = np.argsort(-tezheng_humber)
        #降维矩阵
        self.out_juzhen = tezheng_fuction[:, idx[:self.wei_number]]
        return np.dot(X, self.out_juzhen)

#调用自己写的API函数
pca = PCA_class(wei_number=2)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  #导入数据，维度为4
jiangwei_juzhen_1 = pca.fit_transform(X)
print("自己写的API函数降维的结果")
print(jiangwei_juzhen_1)
#调用API接口
pca = PCA(n_components=2)
jiangwei_juzhen_2 = pca.fit_transform(X)
print("API接口降维的结果")
print(jiangwei_juzhen_2, "\r\n")
print(pca.explained_variance_ratio_)  #输出贡献率
