import math
import numpy as np
import cv2

def bilinear_interpolation(img, dst_h, dst_w):
    src_h, src_w, channels = img.shape
    # 初始化新图像
    new_image = np.zeros([dst_h, dst_w, channels], np.uint8)
    scale_x = dst_h / src_h
    scale_y = dst_w / src_w
    for channel in range(3):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                # 为了实现两个图像的中心点对齐， 需要对每一个进行移动'
                src_x = (dst_x + 0.5) / scale_x - 0.5
                src_y = (dst_y + 0.5)  / scale_y - 0.5
                # 找到dst_x, dst_y 是位于原图像的哪四个点之间 左上，左下， 右上， 右下
                # 只需找到四个点对应的坐标是个点对应的坐标， 这四个点是在原图中
                src_x1 = int(np.floor(src_x))    # 向下取整,--- 浮点数---取整
                src_y1 = int(np.floor(src_y))
                src_x2 = min(src_x1 + 1, src_w - 1)      # 不能直接用向上取整， 还需要考虑最大值的位置
                src_y2 = min(src_y1 + 1, src_h - 1)
                # 计算 对应的点的坐标， 坐标的值row = y, col = x
                ceoff1 = (src_x2 - src_x)  * img[src_y1, src_x1, channel] + (src_x - src_x1)  * img[src_y1, src_x2, channel]   # 需要注意三维坐标的读取方式
                ceoff2 = (src_x2 - src_x)  * img[src_y2, src_x1, channel] + (src_x - src_x1)  * img[src_y2, src_x2, channel]
                new_image[dst_y, dst_x, channel] = int((src_y2 - src_y) * ceoff1 + (src_y - src_y1) * ceoff2)
    return new_image



if __name__ == '__main__':
    path = '/Users/snszz/PycharmProjects/CV/第二周/代码/lenna.png'
    img = cv2.imread(path)
    new_image = bilinear_interpolation(img, 700, 700)
    cv2.imshow('bilinear interpolation', new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 写的过程遇到的细节总结：
"""
1. 中心对齐
src_x = (dst_x + 0.5) / scale_x - 0.5
src_y = (dst_y + 0.5)  / scale_y - 0.5

2. 考虑终点的位置 避免出现索引超出范围
src_x2 = min(src_x1 + 1, src_w - 1)      # 不能直接用向上取整， 还需要考虑最大值的位置
src_y2 = min(src_y1 + 1, src_h - 1)

思路整理：
1. 读取原图的shape
2. 根据原图的通道数，初始化新图
3. 计算新图与原图的比例
4. 对于每一个新图的坐标---如果要实现中心对称---对原图的每一个点 + 0.5， 根据1.，之后找出每个新点对应于原图的那四个点之间，
 x1,y1,x2,y2, 较大的坐标要考虑是否超出索引， 每一个索引为正整数 int,  按照公式计算， 注意小细节，row=y, col = x
"""