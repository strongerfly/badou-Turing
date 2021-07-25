import numpy as np

def warpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 #判断输入点的个数，至少为4个
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) #变换矩阵公式：A * warpMatrix = B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i] = B_i[0]
        B[2*i+1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    #print("乘积后shape", warpMatrix.shape)
    warpMatrix = np.array(warpMatrix).T
    warpMatrix = warpMatrix[0]
    #print("转置后shape", warpMatrix.shape)
    # a33 = 1 #np.insert(arr, obj, values, axis#)
    #arr原始数组，可一可多，obj插入元素位置，values是插入内容，axis是按行按列插入
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = warpPerspectiveMatrix(src, dst)
    print("warpMatrix:", warpMatrix)
