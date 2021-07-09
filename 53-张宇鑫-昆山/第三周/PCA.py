"""PCA实现"""
import numpy
from sklearn.decomposition import PCA

from settings.setting_main import LOG


def PCA_sklearn(src: numpy.ndarray, dst_components: int) -> numpy.ndarray:
    """
    使用sklearn库实现PCA
    :param src: 输入原数据
    :param dst_components:  输出 维度
    :return: pca实现后的 输出
    """
    # 创建一个PCA 对象 是的输出 维度数量为 dst_components
    pcaSrc = PCA(n_components=dst_components)
    # 训练
    pcaSrc.fit(src)
    # 训练 并 输出降维后的 数据
    pca_dst = pcaSrc.fit_transform(src)
    return pca_dst


def PCA_myself(src: numpy.ndarray, dst_components: int) -> numpy.ndarray:
    """
    使用numpy 自己编写方法实现PCA
    :param src: 输入原数据
    :param dst_components:  输出 维度
    :return: pca实现后的 输出
    """

    '''求协方差矩阵'''
    # 求每一列的均值 和 当前列的 差  ===> 中心化
    src_ = src - src.mean(axis=0)
    # 计算协方差矩阵 公式: D = (1/m) * (Z.T) * Z
    cov = numpy.dot(src_.T, src_) / src_.shape[0]
    '''
    使用 numpy.linalg.eig() 计算方形矩阵的特征值和特征向量
    w,v = numpy.linalg.eig(a) 
    :param a: 待求特征值和特征向量的方阵。
    :return: w: 多个特征值组成的一个矢量。备注：多个特征值并没有按特定的次序排列。特征值中可能包含复数。
             v: 多个特征向量组成的一个矩阵。每一个特征向量都被归一化了。第i列的特征向量v[:,i]对应第i个特征值w[i]。
    '''
    w, v = numpy.linalg.eig(cov)

    '''
    获得降序排列特征值的序号
    使用 numpy.argsort  获取 数组值从小到大的索引值
    '''

    w_id_big_to_small = numpy.argsort(-w)  # 为什么 "w" 前 要加个 "-" ??? ===> 因为我们要的是从大到小的索引值,而argsort是获取从小到大

    '''
    根据 dst_components 将数据获得降维矩阵
    '''
    dst_components_ = v[:,w_id_big_to_small[:dst_components]]
    # 降维
    dst = numpy.dot(src_,dst_components_)
    return dst





