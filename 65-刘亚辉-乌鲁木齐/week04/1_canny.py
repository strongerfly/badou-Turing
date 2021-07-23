import numpy as np
import math
import matplotlib.pyplot as plt
import cv2


def gauss_core():
    # 计算高斯核函数
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    return gaussian_filter


def calcu_padding_size(image_rows, image_cols, core_rows, core_cols, step):
    # 计算padding尺寸
    p_row = int(1 / 2 * (step * (image_rows - 1) - image_rows + core_rows))  # 高
    p_col = int(1 / 2 * (step * (image_cols - 1) - image_cols + core_cols))  # 宽
    return p_row, p_col


def convolution_valid(padding_image, image_rows, image_cols, core, core_rows, core_cols, step, p_row, p_col):
    # valid mode(卷积尺寸小于原图)
    new_rows = int((image_rows - core_rows) / step + 1)
    new_cols = int((image_cols - core_cols) / step + 1)
    valid_image = np.zeros((new_rows, new_cols))
    for row in range(valid_image.shape[0]):
        for col in range(valid_image.shape[1]):
            # 计算卷积核移动坐标
            x = row * step + p_row  # 加偏移值
            y = col * step + p_col
            # 灰度图中与卷积核对应的矩阵
            core_image = padding_image[x: x + core_rows, y: y + core_cols]
            # 卷积运算
            convolution_sum = (core_image * core).sum()
            # 运算结果即卷积后的新图像像素值
            valid_image[row, col] = convolution_sum
    return valid_image


def convolution_same(padding_image, image_rows, image_cols, core, core_rows, core_cols, step):
    # same mode(卷积后尺寸与原图相同)
    same_image = np.zeros((image_rows, image_cols))
    for row in range(same_image.shape[0]):
        for col in range(same_image.shape[1]):
            # 计算卷积核移动坐标
            x = row * step
            y = col * step
            # 灰度图中与卷积核对应的矩阵
            core_image = padding_image[x: x + core_rows, y: y + core_cols]
            # 卷积运算
            convolution_sum = (core_image * core).sum()
            # 运算结果即卷积后的新图像像素值
            same_image[row, col] = convolution_sum
    return same_image


def convolution_calculation(image_gray, core, step):
    # 滤波运算
    image_rows, image_cols = image_gray.shape
    # 获取卷积核尺寸
    core_rows, core_cols = core.shape
    ## padding
    # 计算padding尺寸
    p_row, p_col = calcu_padding_size(image_rows, image_cols, core_rows, core_cols, step)
    # 使用 np.pad() 方法对原图进行padding
    padding_image = np.pad(image_gray, ((p_row, p_row), (p_col, p_col)))

    # valid 模式卷积图像
    # valid_image = convolution_valid(padding_image, image_rows, image_cols, core, core_rows, core_cols, step, p_row, p_col)
    # same 模式卷积图像
    same_image = convolution_same(padding_image, image_rows, image_cols, core, core_rows, core_cols, step)
    # 画出与原图的对比图
    return same_image


def calcu_grad(image_gray, sobel_gx, sobel_gy, step):
    # 计算梯度
    image_rows, image_cols = image_gray.shape
    # padding
    # 计算padding尺寸
    p_row, p_col = calcu_padding_size(image_rows, image_cols, sobel_gx.shape[0], sobel_gx.shape[1], step)
    # 使用 np.pad() 方法对原图进行padding
    padding_image = np.pad(image_gray, ((p_row, p_row), (p_col, p_col)))
    # 计算x方向的梯度(差分)
    # same mode(卷积后尺寸与原图相同)
    image_grad_x = np.zeros((image_rows, image_cols))
    image_grad_y = np.zeros((image_rows, image_cols))
    image_grad = np.zeros((image_rows, image_cols))
    for row in range(image_grad_x.shape[0]):
        for col in range(image_grad_x.shape[1]):
            # 计算卷积核移动坐标
            x = row * step
            y = col * step
            # 灰度图中与卷积核对应的矩阵
            core_image = padding_image[x: x + sobel_gx.shape[0], y: y + sobel_gx.shape[1]]
            # 卷积运算
            convolution_sum_x = (core_image * sobel_gx).sum()
            convolution_sum_y = (core_image * sobel_gy).sum()
            # 运算结果即卷积后的梯度值(x, y)
            image_grad_x[row, col] = convolution_sum_x
            image_grad_y[row, col] = convolution_sum_y
            image_grad[row, col] = np.sqrt(image_grad_x[row, col] ** 2 + image_grad_y[row, col] ** 2)
    # 计算梯度方向与X轴的夹角
    image_grad_x[image_grad_x == 0] = 0.00000001  # 避免除数为0
    grad_angle = image_grad_y / image_grad_x
    return image_grad, grad_angle


def nms(image_grad, grad_angle):
    # 非极大值抑制
    img_nms = np.zeros(image_grad.shape)
    for row in range(1, image_grad.shape[0] - 1):
        for col in range(1, image_grad.shape[1] - 1):
            # 找到邻近的八个点
            nearest = image_grad[row - 1:row + 2, col - 1:col + 2]
            # 当前点的梯度角度
            angle = grad_angle[row, col]
            # 根据angle
            if angle <= -1:
                dtmp1 = nearest[0, 1] + 1 / angle * (nearest[0, 1] - nearest[0, 0])
                dtmp2 = nearest[2, 1] + 1 / angle * (nearest[2, 1] - nearest[2, 2])
            elif angle <= 0:
                dtmp1 = nearest[1, 0] - angle * (nearest[0, 0] - nearest[1, 0])
                dtmp2 = nearest[1, 2] - angle * (nearest[2, 2] - nearest[1, 2])
            elif angle <= 1:
                dtmp1 = nearest[1, 2] + angle * (nearest[0, 2] - nearest[1, 2])
                dtmp2 = nearest[1, 0] + angle * (nearest[2, 0] - nearest[1, 0])
            else:
                # elif angle > 1:
                dtmp1 = nearest[0, 1] + 1 / angle * (nearest[0, 2] - nearest[0, 1])
                dtmp2 = nearest[2, 1] + 1 / angle * (nearest[2, 0] - nearest[2, 1])
            if image_grad[row, col] > dtmp1 and image_grad[row, col] > dtmp2:
                img_nms[row, col] = image_grad[row, col]
    return img_nms


def threshold_detect(img_nms, high_th, low_th):
    img_threshold = np.zeros(img_nms.shape)
    for row in range(img_nms.shape[0]):
        for col in range(img_nms.shape[1]):
            # 先判断强边缘点
            if img_nms[row, col] >= high_th:
                img_nms[row, col] = 255  # 在原图中也标注强边缘点，便于后面判断
                img_threshold[row, col] = 255
    # 再判断弱边缘点
    for row in range(1, img_nms.shape[0] - 1):
        for col in range(1, img_nms.shape[1] - 1):
            # 如果是弱边缘点，则判断邻域中是否存在强阈值点
            if low_th < img_nms[row, col] < high_th:
                # 找出邻域
                low_near = img_nms[row - 1:row + 2, col - 1:col + 2]
                for r in range(3):
                    for c in range(3):
                        # 如果邻域内存在强阈值点，则将该弱边缘点标注为强边缘点
                        if low_near[r, c] == 255:
                            img_threshold[row, col] = 255
    return img_threshold


if __name__ == '__main__':
    image_file = 'lenna.png'
    image = cv2.imread(image_file)
    # step 1: gray
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # step 2: 高斯滤波
    # 计算卷积核
    gauss_core = gauss_core()
    # 定义移动窗口的步长
    step = 1
    # 获得高斯滤波后图像
    gauss_image = convolution_calculation(image_gray, gauss_core, step)
    plt.figure(1)
    plt.axis('off')
    plt.imshow(gauss_image.astype(np.uint8), cmap='gray')

    # step 3: 计算梯度(使用差分近似梯度)
    # 利用sobel算子实现
    sobel_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    image_grad, grad_angle = calcu_grad(gauss_image, sobel_gx, sobel_gy, step)
    plt.figure(2)
    plt.axis('off')
    plt.imshow(image_grad.astype(np.uint8), cmap='gray')

    # step 4: NMS非极大值过滤
    image_nms = nms(image_grad, grad_angle)
    plt.figure(3)
    plt.axis('off')
    plt.imshow(image_nms.astype(np.uint8), cmap='gray')

    # step 5: 双阈值检测
    # 设置高低阈值
    low_th = image_grad.mean() * 0.5
    high_th = low_th * 3
    img_threshold = threshold_detect(image_nms, high_th, low_th)
    plt.figure(4)
    plt.axis('off')
    plt.imshow(img_threshold.astype(np.uint8), cmap='gray')
    plt.show()
