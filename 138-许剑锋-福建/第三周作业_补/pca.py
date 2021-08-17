import numpy as np


def pca(matrix, dim):
    print(matrix.shape)
    if len(matrix.shape) != 2:
        return
    mean = np.mean(matrix, axis=0)
    matrix = matrix - mean
    cov = np.cov(matrix)
    eig_val, eig_vec = np.linalg.eig(cov)
    print("eig_val:{}".format(eig_val))
    print("eig_vec:{}".format(eig_vec))
    eigValIndice = np.argsort(eig_val)
    n_eigValIndice = eigValIndice[-1:-(dim + 1):-1]
    n_eigVect = eig_vec[:, n_eigValIndice]
    lowDDataMat = np.dot(n_eigVect.T, matrix)
    print("lowDDataMat.shape:{}".format(lowDDataMat.shape))
    return lowDDataMat


test = np.arange(60)
np.random.shuffle(test)
test = test.reshape((3, 20))
# print(test)
feature = pca(test, 2)
print(feature)

