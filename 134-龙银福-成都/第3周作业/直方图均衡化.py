import cv2
import numpy as np
import matplotlib.pyplot as plt

def Histogram(image):
    gray = np.zeros(256)
    height, width = image.shape
    # 计算灰度直方图
    for i in range(height):
        for j in range(width):
            gray[image[i][j]] += 1
    gray /= (height * width)
    return gray

def Hist_Equalization(gray_image):
    gray = np.zeros(256)
    height, width = gray_image.shape
    # 计算灰度直方图
    for i in range(height):
        for j in range(width):
            gray[gray_image[i][j]] += 1
    # 计算灰度占比
    gray /= (height * width)
    # 计算累计和
    cumsum = np.cumsum(gray)
    equa_t = np.array((256 * cumsum - 1).astype(np.int32))
    # 统计均衡化后的灰度数量，equa_hist为均衡化后的直方图
    equa_hist = np.zeros(256)
    for i in range(256):
        equa_hist[equa_t[i]] += gray[i]
    # 对原灰度图像做均衡化
    for i in range(height):
        for j in range(width):
            gray_image[i][j] = equa_t[gray_image[i][j]]

    return gray_image, equa_hist

if __name__ == '__main__':

    # 1- 使用自定义函数实现均衡化
    image = cv2.imread('lenna.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_hist = Histogram(gray)
    plt.figure()
    plt.subplot(221)
    plt.title("Gray Image")
    plt.imshow(gray, cmap='gray')
    plt.subplot(223)
    plt.title("Gray Image Histogram")
    plt.plot(gray_hist, linewidth=1)
    # plt.hist(gray.ravel(), 256)

    equa_image, equa_image_hist = Hist_Equalization(gray)
    plt.subplot(222)
    plt.title("Histogram Equalization")
    plt.imshow(equa_image, cmap='gray')
    plt.subplot(224)
    plt.title("Histogram Equalization Hist")
    plt.plot(equa_image_hist, linewidth=1)
    # plt.hist(equa_image.ravel(), 256)

    plt.tight_layout() # 设置默认的间距
    plt.show()


    """
    # 2- 使用 cv2 和 matplotlib 实现均衡化及可视化
    image = cv2.imread('lenna.png', 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    dst = cv2.equalizeHist(gray)
    # 计算直方图
    grayHist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    dstHist = cv2.calcHist([dst], [0], None, [256], [0, 256])
    # 画直方图
    plt.figure()
    plt.subplot(211)
    plt.title("Original Histogram")
    plt.plot(grayHist, linewidth=1)
    plt.subplot(212)
    plt.title("Equalize Histogram")
    plt.tight_layout()
    plt.plot(dstHist)
    plt.show()
    # 显示原图和均衡化后的图像
    cv2.imshow("Origin --- Equalization", np.hstack([gray, dst]))
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    """
    # 3- 彩色图像直方图均衡化
    image = cv2.imread('lenna.png', 1)
    cv2.imshow('Origin', image)
    # 分离 3 个通道的数据
    (b, g, r) = cv2.split(image)
    # 对每个通道进行均衡化
    eb = cv2.equalizeHist(b)
    eg = cv2.equalizeHist(g)
    er = cv2.equalizeHist(r)
    # 合并 3 个通道
    dst = cv2.merge((eb, eg, er))
    # 显示图像
    cv2.imshow("Equalization", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """