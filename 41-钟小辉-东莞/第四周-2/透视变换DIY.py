
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':

    pic_path = 'photo1.jpg'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.jpg':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值就是灰度化了

    width = img.shape[1]
    height = img.shape[0]
    plt.figure()
    plt.imshow(img,cmap='gray')


    moving_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    moving_points = np.array(moving_points)

    fixed_points = np.float32([[0,0],[500,0],[0,400],[500,400]])
    fixed_points = np.array(fixed_points)

    #1.获取转换矩阵
    A = WarpPerspectiveMatrix(moving_points,fixed_points)
    # print(A)

    #原图片的四个角点
    X = [0,width-1,0,width-1]
    Y = [0,0,height-1, height-1]
    Z = np.ones(len(X))
    moving_points_mat = np.array([X[:],Y[:],Z[:]]).astype(float)
    print(moving_points_mat)

    #2.转换后
    dst_points = A.dot(moving_points_mat)
    print(dst_points)
    # print(dst_points.shape)

    # for i in range(dst_points.shape[1]-1):
    #     dst_points[0:1, i] = dst_points[0:1, i] / dst_points[2, i]
    for i in range(dst_points.shape[1]):
        dst_points[0:2, i] = dst_points[0:2, i] / dst_points[2, i]
    print(f"目标点位\n{dst_points.astype(float)}")

    plt.figure()
    plt.scatter(dst_points[0,:], dst_points[1,:])
    plt.show()

    # 透视变换后图像逐像素进行插值赋值
    min_x = min(dst_points[0,:])
    max_x = max(dst_points[0,:])
    min_y = min(dst_points[1,:])
    max_y = max(dst_points[1,:])

    W = round(max_x - min_x)
    H = round(max_y - min_y)
    wrapImg = np.zeros((H, W))
    A = np.mat(A)

    temp_point = []
    for i in range(H-1):
         for j in range(W-1):
    # for i in range(1):
    #      for j in range(1):

            x = min_x + j  #使得x，y的范围在原坐标范围内
            y = min_y + i

            #111
            # moving_point = (np.array([x, y, 1]).dot(A.I)).T
            B = np.array([[x], [y], [1]])
            moving_point = A.I * B


            for ii in range(moving_point.shape[0]-1):
                temp_point =[moving_point[0]/moving_point[2],moving_point[1]/moving_point[2]]
            # print(temp_point)

            if temp_point[0] >= 0 and (temp_point[0] < width-1)and (temp_point[1] >= 0) and (temp_point[1] < height-1):
                X1 = np.round(temp_point[1])
                Y1 = np.round(temp_point[0])
                wrapImg[i, j] = img[X1.astype(int),Y1.astype(int)]
                # print(wrapImg)

    plt.figure()
    # plt.imshow(wrapImg.astype(np.uint8),cmap='gray')
    plt.imshow(wrapImg.astype(np.uint8))
    plt.show()


