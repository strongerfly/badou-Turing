import numpy as np
import cv2

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1,
                       0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0,
                           A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)  # 创建矩阵
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32  A.I：求逆矩阵

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]  # 8*1 =》 1*8转换形态
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def cv_imshow(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':


    img = cv2.imread('photo1.jpg')
    img_copy = img.copy()

    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[100, 100], [437, 100], [100, 588], [437, 588]])
    print(img.shape)

    # 生成透视变换矩阵m
    m = WarpPerspectiveMatrix(src, dst)
    print("生成透视变换矩阵:{}".format(m))
    result = cv2.warpPerspective(img_copy, m, (537, 688))

    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)