import numpy as np
from sklearn.decomposition import PCA
# 导入数据，维度为4
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
# 降到2维,初始设置
pca = PCA(n_components=2)
# 训练
pca.fit(X)
# 降维后的数据
newX=pca.fit_transform(X)
PCA(copy=True, n_components=2, whiten=False)
# 输出贡献率
print(pca.explained_variance_ratio_)
# 输出降维后的数据
print(newX)
