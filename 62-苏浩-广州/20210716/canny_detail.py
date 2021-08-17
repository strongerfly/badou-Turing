import numpy as np
import matplotlib.pyplot as plt
import math

'''
脚本直接执行，被其他脚本调用将不用被执行
if __name__ == '__main__':
Canny算子
1. 对图像进行灰度化
2. 对图像进行高斯滤波：
根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
4 对梯度幅值进行非极大值抑制
5 用双阈值算法检测和连接边缘
'''

if __name__ == '__main__':
    pic_path = 'lenna.png' 
    # 读入的范围是0-1；
    # 区别于：对于cv2.imread()函数来说，读取出来的值为BGR，且范围是0-255
    img = plt.imread(pic_path)
    print(img)

    # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算，这里是进行字符串判断
    if pic_path[-4:] == '.png':
        # 还是浮点数类型
        img = img * 255
    # 取均值就是灰度化了
    # axis=-1/1计算每一行的均值
    # axis=0计算每一列的均值
    img = img.mean(axis=-1)
    print("==============================")
    print(img)
 
    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5
    # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    # ？高斯分布，宽度一般取标准差的3倍？这里可能也是为了凑出5*5这个比较常用的trade off，一般5*5的效果不错
    dim = int(np.round(6 * sigma + 1))
    # 最好是奇数,不是的话加一
    if dim % 2 == 0:
        dim += 1
    # 存储高斯核，这是数组，dim*dim的一个全0数组
    Gaussian_filter = np.zeros([dim, dim])
    # 生成一个序列,59行使用
    # tmp = [i-5//2 for i in range(5)]---[-2, -1, 0, 1, 2]
    tmp = [i-dim//2 for i in range(dim)]
    # 计算高斯核
    n1 = 1/(2*math.pi*sigma**2)
    # 55行使用，当作e幂次方的分母
    n2 = -1/(2*sigma**2)
    # 根据G（x,y）公式，计算出每个（x，y）位置的值
    for i in range(dim):
        for j in range(dim):
            # 转换为高斯核的公式
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    # 归一化，使每个坐标的值，计算总和为1
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # 计算图像的长度和宽度的尺寸
    dx, dy = img.shape
    # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    # 新建立一个图像容器
    img_new = np.zeros(img.shape)
    # //在python中表示整数除法
    tmp = dim//2
    # 边缘填补
    # 这里填补了2层
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            # 这部分是高斯核计算，填补了两层的图像和高斯核之间的卷积
            # 得到高斯滤波后的图像数据集合
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    #
    #
    # 这里显示的是原图像灰度处理后图像数据
    plt.figure(1)
    # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #
    #
    #
    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    # 检测水平，垂直，对角边缘（sobel）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 存储梯度图像
    # 高斯滤波后的图像尺寸
    # 这里是512*512
    img_tidu_x = np.zeros(img_new.shape)
    print(img_new.shape)
    # dx dy原图像的尺寸
    # 这里512*512
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    # 边缘填补，根据上面矩阵结构所以写1
    # 继续填补了一层？为什么
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            # 填补一层后的矩阵，进行卷积计算
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
            # 这里有又是为什么？这样既经历了sobel x方向又经历了sobel y方向两个方向的检测
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y/img_tidu_x
    #
    #
    # 梯度后图像（经历灰度化，高斯，sobel，梯度）
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #
    #
    #
    # 3、非极大值抑制
    # 实现非极大值就置零的情况
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            # 在8邻域内是否要抹去做个标记
            flag = True
            # 梯度幅值的8邻域矩阵
            temp = img_tidu[i-1:i+2, j-1:j+2]

            # 使用线性插值法判断抑制与否
            # 需要进一步理解线性插值
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
                    # false算是置0情况
            # 下面部分是不置为0的情况，因为一开始创建的就是0的维度图像数据
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    #
    #
    # 非极大值抑制之后的数据图像
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    #
    #
    #
    # 4、双阈值检测，连接边缘。
    # 遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    # 就是上一步的过程中为true的点
    lower_boundary = img_tidu.mean() * 0.5
    # 这里我设置高阈值是低阈值的三倍
    high_boundary = lower_boundary * 3
    zhan = []
    # 外圈不考虑了，所以是1到img_yizhi.shape[0]-1的情况
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1]-1):
            # 取，一定是边的点
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255 # 使之为255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0
    # 对于进栈不为零，后续部门疑似点处理
    #需要进一步理解
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
 
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
 
    #
    #
    # 经历完整的Canny算法（高斯滤波，）的最终结果
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # 关闭坐标刻度值
    plt.axis('off')
    # 绘图
    plt.show()
