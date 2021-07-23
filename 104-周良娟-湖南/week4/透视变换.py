import cv2
import numpy as np
def warpmatrix(src, dst):
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A * warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):  # 矩阵的构造
        A[2*i, :] = [A[i, 0], A[i,1], 1, 0, 0, 0,
                     -A[i,0]*B[i, 0], -A[i,1]*B[i, 0]]
        B[2*i] = B[i,0]

        A[2*i+1, :] = [0, 0, 0, A[i,0], A[i,1], 1,
                       -A[i,0]*B[i, 1], -A[i,1]*B[i, 1]]
        B[2*i+1] = B[i, 1]

    A = np.mat(A)
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    # insert(arr, obj, values, axis=None), 分别是arr, 插入的位置， 插入的值
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

# if __name__ == '__main__':
#     print('warpMatrix')
#     src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
#     src = np.array(src)
#
#     dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
#     dst = np.array(dst)
#
#     warpMatrix = warpmatrix(src, dst)
#     print(warpMatrix)

if __name__ == '__main__':
    new_img = cv2.imread('computer.png', 1)
    print(new_img.shape, 'new')
    result3 = new_img.copy()
    src = np.float32([[208,10], [1121, 5], [332, 983], [1121, 856]])
    dst = np.float32([[0, 0], [1121, 0], [0, 993], [1121, 993]])
    m = cv2.getPerspectiveTransform(src, dst)
    print(m, 'swarmmatrix')
    result = cv2.warpPerspective(result3, m, (1121, 993))
    cv2.imshow('src', new_img)
    cv2.imshow('result', result)
    cv2.waitKey(0)






    # method2

