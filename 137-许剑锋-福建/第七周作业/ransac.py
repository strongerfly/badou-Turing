import cv2
import numpy as np
import matplotlib.pyplot as plt


def linear_fit(data):
    if len(data.shape) != 2:
        print("数据长度不一致")
        return

    data_x = data[0]
    data_y = data[1]
    size = len(data_x)
    # 计算k值
    sum_x = np.sum(data_x)
    sum_y = np.sum(data_y)
    sum_x_square = np.sum([x ** 2 for x in data_x])
    sum_xy = np.sum([data_x[i] * data_y[i] for i in range(size)])
    k = (size * sum_xy - sum_y * sum_x) / (size * sum_x_square - sum_x ** 2)
    b = (sum_y / size) - k * (sum_x / size)
    return k, b


def ransac_linear(data, k, d):
    '''

    :param data: 样本点
    :param k: 最大迭代次数
    :param d: 正确点的距离阈值
    :return: best_fit 最优拟合解（返回null，未找到）
    '''
    best_fit = None
    if len(data.shape) != 2:
        print("数据长度不一致")
        return best_fit
    data_x = data[0]
    data_y = data[1]
    correct_max = 0
    for i in range(k):
        data_copy = data[:]
        random_position = np.arange(0, len(data_x))
        np.random.shuffle(random_position)
        fit_dot = data_copy[:, random_position[0:k]]
        fit_k, fit_b = linear_fit(fit_dot)
        # 计算范围内的个数
        compute_y = fit_k * data_x + fit_b
        compute_error = (data_y - compute_y) ** 2
        correct_num = np.sum(compute_error < d)
        if best_fit is None:
            best_fit = np.array([fit_k, fit_b])

        elif correct_num > correct_max:
            correct_max = correct_num
            best_fit = np.array([fit_k, fit_b])

    return best_fit


data = np.random.normal(0, 10, 100).reshape((2, 50))
# print(data)

# k, b = linear_fit(data)
# print("k:{}".format(k))
# print("b:{}".format(b))
# plt.scatter(data[0], data[1])
# plt.show()
k, b = ransac_linear(data, 10, 50)
print("k:{},b:{}".format(k, b))
x = data[0]

y = k * x + b

plt.title("ransac")
plt.plot(x, y)
plt.scatter(data[0], data[1])
plt.show()