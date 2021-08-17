import numpy as np
import scipy as sp
import scipy.linalg as sl

""" 在数据中随机选择 n 个点设定为内群 """
def random_partition(n, n_data):
    all_idxs = np.arange(n_data)  # 获取n_data的下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引 -> 为了随机选择 n 个数据点设定为内群
    idxs1 = all_idxs[:n]  # 取出前 n 个索引值列表
    idxs2 = all_idxs[n:]  # 取出 n 以后的索引值列表
    return idxs1, idxs2

""" 最小二乘求线性解, 用于RANSAC的输入模型 """
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False): # 初始化
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    # 对数据进行最小二乘线性拟合
    def fit(self, data):    # 拟合, data 是选定的内群数据点, 是一个列表
        # np.vstack 按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T   # 输入列：第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 输出列：第二列Yi-->行Yi
        # 生成模型：y_fit = x[0] + x[1] * x_fit
        x, residues, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    # 计算均方误差
    def get_error(self, data, model): # data 其他没选到的数据, 把其他没选到的点带入刚才建立的模型中, 计算是否为内群
        A1 = np.vstack([data[:, i] for i in self.input_columns])
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # i=[0] 第一列Xi-->行Xi    A.shape=[1, 450]
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # i=[1] 第二列Yi-->行Yi   B.shape=[1, 450]
        B_fit = sp.dot(A, model)  # 计算的y_fit值, B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # sum squared error per row
        return err_per_point

""" 随机采样一致性(random sample consensus)思想 """
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）
    """
    iterations = 0  # 迭代次数的变量
    bestfit = None  # 最优拟合结果
    besterr = np.inf  # 设置默认值 np.inf(-inf,inf):infinity,inf表示正无穷,-inf表示负无穷
    best_inlier_idxs = None # 最优的内群点索引
    while iterations < k:  # 未满足最大迭代次数则继续迭代
        # 1-在数据中随机选择 n 个点设定为内群
        # maybe_idxs代表提前设定为内群的数据点索引(50个), test_idxs代表用于后续测试的数据点索引(450个)
        maybe_idxs, test_idxs = random_partition(n, data.shape[0]) # n=50, data.shape=[500, 2]
        print('test_idxs = ', test_idxs)  # 打印用于测试的数据点索引
        maybe_inliers = data[maybe_idxs, :]  # 取出设为内群的数据(Xi,Yi)存入 maybe_inliers -> 内群点
        test_points = data[test_idxs]  # 取出用于测试的若干行数据点(Xi,Yi)放入 test_points

        # 2-计算适合内群的模型 -> 求模型的参数
        maybemodel = model.fit(maybe_inliers)  # 拟合模型：y_fit = maybemodel[0] + maybemodel[1] * x_fit

        # 3-把其他没选到的数据点带入刚才建立的模型中, 计算是否为内群
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        print('test_err = ', test_err < t) # test_err小于阈值设为1(True), 大于阈值设为0(False)
        also_idxs = test_idxs[test_err < t] # 取出没选到的数据点且满足模型内群点的索引存入also_idxs
        print('also_idxs = ', also_idxs) # 打印没选到的数据点且满足模型内群点的索引
        also_inliers = data[also_idxs, :] # 取出没选到但满足模型内群点的数据点放入also_inliers
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))

        # 4-记下内群点数量
        print('d = ', d)
        # len(also_inliers)代表模型选定的内群点数量
        if (len(also_inliers) > d):
            # np.concatenate()按轴axis(默认为0)连接array组成一个新的array
            # maybe_inliers为提前选定的50个建立初始模型的内群数据点(Xi, Yi)
            # also_inliers为满足建立的模型且初始未选中的数据点
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接, betterdata为两部分数据点的连接
            bettermodel = model.fit(betterdata) # 将两部分数据点放在一起再次进行拟合, 获得更好的模型
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr: # besterr为正无穷
                bestfit = bettermodel # 最优的模型拟合参数
                besterr = thiserr # 平均误差
                # best_inlier_idxs为选出的最优内群点的索引,包括预先选定的和满足模型的内群点
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    # == 是比较两个对象的内容是否相等，即两个对象的“值“是否相等，不管两者在内存中的引用地址是否一样
    # is 比较的是两个实例对象是不是完全相同，它们是不是同一个对象，占用的内存地址是否相同. 即is比较两个条件：1-内容相同; 2-内存中地址相同
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit # 返回模型最优拟合解


def test():
    # 生成理想数据
    n_samples = 500     # 样本个数
    n_inputs = 1        # 输入变量个数
    n_outputs = 1       # 输出变量个数
    # np.random.random((1000, 20))上面这个就代表生成1000行20列的浮点数，浮点数都是从 0-1 中随机
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量, 形状为(500, 1)
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率（只输出一个值，服从标准正态分布）
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x(500x1) * k(1*1) -> y(500x1)

    # 加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加100个"局外点" -> 把500个数据点中的100个设为局外点
        n_outliers = 100    # 局外点数量
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将索引值 all_idxs 的顺序打乱
        outlier_idxs = all_idxs[:n_outliers]  # 获取100个0-500的随机局外点索引
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model -> np.hstack():横向拼接，增加特征量
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:[0]
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:[1]
    debug = False

    # 方法1-最小二乘线性回归拟合数据 -> 模型：y_fit = linear_fit[0] + linear_fit[1] * x
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # 方法2-随机采样一致性ransac思想求解线性模型参数
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    # all_data 样本点, model 假设模型(类实例):事先自己确定, 50 生成模型所需的最少样本点, 1000 最大迭代次数
    # 7e3 阈值:作为判断点满足模型的条件, 300 拟合较好时,需要的样本点最少的个数,当做阈值看待
    # ransac_fit为模型最优拟合解, ransac_data为500个数据中被设定为内群点的索引

    if 1:
        import pylab
        # numpy.argsort()函数返回的是数组值从小到大的索引值, 如x=np.array([3,1,2]),返回的数组x=[1 2 0]
        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs] # 把A_exact(x)从小到大排序存入A_col0_sorted

        if 1:
            # 画所有待拟合数据点的散点图
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            # 画内群点的散点图, ransac_data['inliers']是前后所有选中的内群点的索引
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        # 画ransac拟合得到的模型直线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        # 用产生数据的斜率画初始直线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        # 用所有数据进行最小二乘拟合得到的直线
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()


if __name__ == "__main__":
    test()