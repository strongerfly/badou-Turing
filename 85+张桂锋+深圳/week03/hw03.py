import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

'''
第三周作业：
1）直方图均衡化实现
2）卷积原理
3）PCA实现
'''
# 直方图均衡化实现
def Hist_equalize(img):
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    return result


# 卷积原理
def conv_fun(img):
    # 定义卷积核（3*3）
    k = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    print(k.shape)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img_gray[1:3,1:3])
    n,m=img_gray.shape
    conv = np.zeros((n-2, m-2), dtype='uint8')
    for i in range(n-2):
        for j in range(m-2):
            conv[i][j] =(img_gray[i:i + 3,j:j + 3]*k).sum()
            if conv[i][j]<0:
                conv[i][j]=0
            elif conv[i][j]>255:
                conv[i][j] = 255
    print(conv)
    return conv

class CPCA(object):
    def __init__(self, X, K):
        '''
        :param X,训练样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得
    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #样本集的特征均值
        print('样本集的特征均值:\n',mean)
        centrX = self.X - mean ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        #样本集的样例总数
        ns = np.shape(self.centrX)[0]
        #样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    #直方图均衡化实现
    Hist_equalize_img=Hist_equalize(img)
    cv2.imshow("dst_rgb", Hist_equalize_img)
    #卷积原理
    conv=conv_fun(img)
    cv2.imshow("conv", conv)
    #通过sk-learn包实现
    X = np.array(
        [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    print(X)  # 输出降维后的数据
    print(pca.explained_variance_ratio_)  # 输出贡献率
    print(newX)  # 输出降维后的数据
    #通过公式实现
    K = np.shape(X)[1] - 1
    pca = CPCA(X,K)
    cv2.waitKey(0)