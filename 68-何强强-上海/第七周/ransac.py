# -*- coding:utf-8 -*-
"""
# Linear Least Squares
lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
          check_finite=True, lapack_driver=None):
Returns
-------
x : (N,) or (N, K) ndarray
    Least-squares solution.  Return shape matches shape of `b`.
residues : (K,) ndarray or float
    Square of the 2-norm for each column in ``b - a x``, if ``M > N`` and
    ``ndim(A) == n`` (returns a scalar if b is 1-D). Otherwise a
    (0,)-shaped array is returned.
rank : int
    Effective rank of `a`.
s : (min(M, N),) ndarray or None
    Singular values of `a`. The condition number of a is
    ``abs(s[0] / s[-1])``.
"""
import numpy as np
import pylab
import scipy as sp
import scipy.linalg as sl


def random_partition(n, data_count):
    all_index = np.arange(data_count)
    np.random.shuffle(all_index)
    return all_index[:n], all_index[n:]


def ransac(data, model, n, max_iter, err_threshold, d):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        max_iter - 最大迭代次数
        err_threshold - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        best_model - 最优拟合解（返回nil,如果未找到）
        best_inline_index - 对应内群点
    """
    # 迭代次数
    iteration = 0
    best_model, best_inline_index, min_err = None, None, np.inf
    while iteration < max_iter:
        # 拆分训练集
        inline_index, outline_index = random_partition(n, data.shape[0])
        print("inline_index", inline_index)
        print("outline_index", outline_index)
        inline_data, outline_data = data[inline_index, :], data[outline_index]
        # 使用随机数据生成一个模型（带参数）
        maybe_model = model.fit(inline_data)
        # 计算外群点相对于当前模型的误差值
        test_errs = model.get_error(outline_data, maybe_model)
        # 误差小于阈值，则该点也属于内群
        maybe_index = outline_index[test_errs < err_threshold]
        maybe_inline_data = data[maybe_index, :]
        if len(maybe_inline_data) + n > d:
            # 拼接获取所有内群点
            all_inline_data = np.concatenate((inline_data, maybe_inline_data))
            better_model = model.fit(all_inline_data)
            better_model_err = model.get_error(all_inline_data, better_model)
            mean_err = np.mean(better_model_err)
            if mean_err < min_err:
                best_model, min_err = better_model, mean_err
                best_inline_index = np.concatenate((inline_index, maybe_index))
        iteration += 1

    if best_model is None:
        raise ValueError("did't meet fit acceptance criteria")
    return best_model, best_inline_index


class LinearLeastSquareModel:
    """ 最小二乘线性求解 """
    def __init__(self, input_columns, output_columns):
        """

        :param input_columns:  参数列
        :param output_columns: 输出结果列
        """
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, data):
        a = np.vstack([data[:, i] for i in self.input_columns]).T
        b = np.vstack([data[:, i] for i in self.output_columns]).T
        x, residues, rank, s = sl.lstsq(a, b)
        return x

    def get_error(self, data, model):
        """返回残差平方和"""
        a = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        b = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        b_fit = np.dot(a, model)  # 计算的y值,b_fit = model.k*a + model.b
        return np.sum((b - b_fit) ** 2, axis=1)


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    a_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    b_exact = np.dot(a_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声,最小二乘能很好的处理
    a_noisy = a_exact + np.random.normal(size=a_exact.shape)  # 500 * 1行向量,代表Xi
    b_noisy = b_exact + np.random.normal(size=b_exact.shape)  # 500 * 1行向量,代表Yi

    # 添加"局外点"
    n_outliers = 100
    all_idxs = np.arange(a_noisy.shape[0])  # 获取索引0-499
    np.random.shuffle(all_idxs)  # 将all_idxs打乱
    outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
    a_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi
    b_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi
    # setup model
    all_data = np.hstack((a_noisy, b_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    ransac_fit, inline_indexes = ransac(all_data, model, 50, 1000, 7e3, 300)

    sort_idxs = np.argsort(a_exact[:, 0])
    A_col0_sorted = a_exact[sort_idxs]  # 秩为2的数组

    pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(a_noisy[inline_indexes, 0], b_noisy[inline_indexes, 0], 'bx',
               label="RANSAC data")

    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, perfect_fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    test()
