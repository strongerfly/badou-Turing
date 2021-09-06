import cv2
import numpy as np
'''
cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
参数解释：

InputArray src	输入图片
OutputArray dst	输出图片
Size	输出图片尺寸
fx, fy	沿x轴，y轴的缩放系数
interpolation	插入方式
interpolation 选项所用的插值方法：
INTER_NEAREST  最近邻插值
INTER_LINEAR   双线性插值（默认设置）
INTER_AREA     使用像素区域关系进行重采样。
INTER_CUBIC    4x4像素邻域的双三次插值
INTER_LANCZOS4 8x8像素邻域的Lanczos插值

'''
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    print(img.shape, 'img.shape')
    # 1. 图片变成8*8
    img_shrink = cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('img_shrink lenna', img_shrink)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 2. 灰度化
    gray = cv2.cvtColor(img_shrink, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_shrink_gray lenna', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 3. 求平均值：计算灰度图所有像素的平均值。
    all_ipx_mean = np.mean(gray)
    # 4.比较：像素值大于平均值记作1，相反记作0，总共64位。
    new = gray.copy()
    new[new < all_ipx_mean] = 0
    new[new >= all_ipx_mean] = 1
    hash1 = ''
    for i in range(8):
        for j in range(8):
            hash1 += str( new[i,j] )
    print(hash1,"hash1")








