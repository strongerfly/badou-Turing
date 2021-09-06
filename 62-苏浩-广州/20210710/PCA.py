import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris

# �������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
x, y = load_iris(return_X_y=True)
# ����pca�㷨�����ý�ά�����ɷ���ĿΪ2
pca = dp.PCA(n_components=2)
# ��ԭʼ���ݽ��н�ά��������reduced_x��
reduced_x = pca.fit_transform(x)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
# ���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
