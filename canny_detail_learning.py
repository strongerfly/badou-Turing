import  numpy as np
import matplotlib.pyplot as plt
import cv2
import math

#matplotlib.pyplot、PIL、cv2三种库，这三种库图像读取的区别，说明见下""""""
"""
用python进行图像处理中分别用到过matplotlib.pyplot、PIL、cv2三种库，这三种库图像读取和保存方法各异，并且图像读取时顺序也有差异，
如plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序。使用时需要倍加注意。
。对这三种库图像读取保存进行梳理。与原参考资源有一定差异。当前使用为python3.5版本。
读取图像
1.matplotlib.pyplot
matplotlib读取进来的图片是unit8,0-255范围。
2.PIL.image.open
PIL是有自己的数据结构的，但是可以转换成numpy数组，转换后的数组为unit8，0-255
3.cv2.imread
opencv读进来的是numpy数组，类型是uint8，0-255。
4.plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序。使用时需要倍加注意。
显示图像
均用plt.imshow(img)：因为opencv读取进来的是bgr顺序呢的，而imshow需要的是rgb顺序，因此需要先反过来plt.imshow(img[..., -1::-1])。
保存图像
1 PIL.image - 保存PIL格式的图片
img.save("1.jpg")
2.cv2.imwrite - 保存numpy格式的图片
cv2.imwrite("1.jpg")

"""

if __name__ == '__main__':
    pic_path = r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png"

    #1.图像灰度化：说明见下""""""
    """
    2.1 对原始图像进行灰度化

        Canny算法通常处理的图像为灰度图，因此如果摄像机获取的是彩色图像，那首先就得进行灰度化。对一幅彩色图进行灰度化，就是根据图像各个通道的采样值进行加权平均。以RGB格式的彩图为例，通常灰度化采用的方法主要有：

        方法1：Gray=(R+G+B)/3;

        方法2：Gray=0.299R+0.587G+0.114B;（这种参数考虑到了人眼的生理特点）

        注意1：至于其他格式的彩色图像，可以根据相应的转换关系转为RGB然后再进行灰度化；

        注意2：在编程时要注意图像格式中RGB的顺序通常为BGR。
    """

    #plt.imread和PIL.Image.open读入的都是RGB顺序，而cv2.imread读入的是BGR顺序
    img = plt.imread(pic_path)

    """比较numpy.array不同,除了变成了浮点数外，其实只是rgb顺序不同
    img = img * 255  # 还是浮点数类型
    print(img[0:1])
    img2 = cv2.imread(pic_path)
    print(img2[0:1])
    
    如果显示，与CV2也是不同的
    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # ax.axis('off')
    # plt.title('matplotlib.pyplot.imread() function Example',
    #           fontweight="bold")
    # plt.show()
    """

    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型

    img = img.mean(axis=-1)  # 取均值就是灰度化了,Gray=(R+G+B)/3，每个像素点等于三通道的点求平均

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # round是四舍五入函数，根据标准差求高斯核是几乘几的，也就是维度
    if dim % 2 == 0:  # 最好是奇数,不是的话加一
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了,5*5
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列[-2,-1,0,1,2] ，均值为0

    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补

    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    # plt.show()

    # 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)

    #如果有为0的赋值为0.00000001
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    """
    梯度角度θ范围从弧度-π到π，然后把它近似到四个方向，分别代表水平，垂直和两个对角线方向（0°,45°,90°,135°）
    可以以±iπ/8（i=1,3,5,7）分割，落在每个区域的梯度角给一个特定值，代表四个方向之一
    """
    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵

            # 参考https://www.jianshu.com/p/d21a33a7901a
            # 参考图片1.png 2.png
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                # #tan(135°) = -1/tan(91°) = -57.28996/tan(271°) = -57.28996/tan(315°) = -1
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]

                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                # tan(45°) = 1/ tan(90°) = ∞ /tan(225°) = 1/ tan(270°) = ∞
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]

                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                # tan(0°) = 0 /tan(45°) = 1 tan(181°) = 0.01746 /tan(225°) = 1
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]

                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                # tan(150°) = -0.57735 /tan(135°) = -1/tan(316°) = -0.96569 /tan(360°) = 0
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]

                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # print(img_tidu)
    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
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

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()