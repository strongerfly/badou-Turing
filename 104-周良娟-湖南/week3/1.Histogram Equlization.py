import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_equlization(img, p):
    """
    img:原始图像
    p: 直方图大小，一般等于灰度级数
    q: 均衡化后直方图大小，
    """
    h, w = img.shape
    new_img = np.zeros([h, w], img.dtype)
    sumPi = 0
    equliz_hist = np.zeros(p)  # 初始化
    img_hist = cv2.calcHist([img], [0], None, [p], [0, p]).ravel()
    img_hist = img_hist / float(h * w)
    for i in range(p):
        sumPi += img_hist[i]
        qi = int(round(sumPi * 256 - 1))
        if qi < 0:
            qi = 0
        if qi > 256:
            qi = 255
        equliz_hist[i] = qi
    equliz_hist.astype(np.int64)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = equliz_hist[img[i,j]]

    return equliz_hist, new_img


if __name__ == '__main__':
    img = np.array([[1,3,9,9,8],
                    [2,1,3,7,3],
                    [3,6,0,6,4],
                    [6,8,2,0,5],
                    [2,9,2,6,0]], dtype='uint8')
    equliz_hist, new_img = hist_equlization(img, p=10)
    print('直方图均衡后对应的值: \n', equliz_hist)
    print('原矩阵: \n', img)
    print('均衡直方图矩阵: \n', new_img)
    print('cv2对应的直方图均衡函数: \n', cv2.equalizeHist(img))


    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('Original Histogram')
    plt.xlabel('BIns')
    plt.ylabel('# of Pixels_Original')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

    # 直方图均衡化之后的展示
    hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('Equlization Histogram')
    plt.xlabel('BIns')
    plt.ylabel('# of Pixels_Equlization')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()



# cv2对应的直方图均衡函数， 但是结果和我直接写的代码有些不同，
# equ = cv2.equalizeHist(img)
# equ

# lenna图像
img_lenna = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img_lenna, cv2.COLOR_BGR2GRAY)
# cv2.equlizeHist(gray)
dst = cv2.equalizeHist(gray)

# 比较原图的hist与均衡化之后的结果
img_hist = cv2.calcHist([img_lenna], [0], None, [256], [0, 256])
dst_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('BIns')
plt.ylabel('# of Pixels')
plt.plot(dst_hist)
plt.plot(img_hist)
plt.xlim([0, 256])
plt.show()

# 多通道的结果要注意 分通道处理， 之后合并
img = cv2.imread('lenna.png', 1)

# 分通道均衡化
(b,g,r) = cv2.split(img)

bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

# 合并
result = cv2.merge((bh, gh, rh))
# 展示
cv2.imshow('dst_rgb', result)
cv2.waitKey(0)
cv2.destroyAllWindows()