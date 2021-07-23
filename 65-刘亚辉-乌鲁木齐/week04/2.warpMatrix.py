import numpy as np
import matplotlib.pyplot as plt
import cv2


def warp_matrix(src, dst):
	# num为点的个数
	num = src.shape[0]
	# 构造AB矩阵
	A = np.zeros((2*num, 8))
	B = np.zeros((2*num, 1))

	# 根据推导公式中的关系求出AB
	for i in range(num):
		src_i = src[i, :]
		dst_i = dst[i, :]
		A[2*i, :] = [src_i[0], src_i[1], 1, 0, 0, 0, -src_i[0]*dst_i[0], -src_i[1]*dst_i[0]]
		B[2*i] = dst_i[0]
		A[2*i+1, :] = [0, 0, 0, src_i[0], src_i[1], 1, -src_i[0]*dst_i[1], -src_i[1]*dst_i[1]]
		B[2*i+1] = dst_i[1]
	# 转为矩阵
	A = np.mat(A)
	# 求出w
	w = A.I * B
	# 在w末尾插入1后，将其转为3*3数组，得到最终的w
	w = np.array(w)
	w = np.insert(w, w.shape[0], values=1, axis=0)
	return w.reshape((3, 3))


# 原图为鱼眼摄像头拍摄图像
fisheye_image = 'OIP-C.jpg'
image = plt.imread(fisheye_image)
plt.subplot(121)
plt.imshow(image, cmap='gray')

# # 将原图的四角拉直
src = np.array([[0, 97], [0, 374], [351, 374], [351, 85]])
dst = np.array([[0, 0], [0, 474], [351, 474], [351, 0]])
warp = warp_matrix(src, dst)
image_trans = cv2.warpPerspective(image, warp, (351, 474))
plt.subplot(122)
plt.imshow(image_trans, cmap='gray')
plt.show()
