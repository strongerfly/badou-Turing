# 第二周作业 1.最邻近插值算法实现 2.双线性插值算法实现 3.rgb2gray
import cv2
import numpy


# 1.最邻近插值
def near_inter(src_img, target):
    sh, sw, sc = src_img.shape
    th, tw = target[:2]
    tag_img = numpy.zeros((th, tw, sc), numpy.uint8)  # 目标图空矩阵
    rh, rw = th / sh, tw / sw  # 目标图和原图宽高比例
    for i in range(th):
        for j in range(tw):
            x = int(i / rh)  # 原图邻近点高坐标
            y = int(j / rw)  # 原图邻近点宽坐标
            tag_img[i, j] = src_img[x, y]  # 将原图邻近点的像素赋值给目标图
    return tag_img


# 2.双线性插值
def bilinear_inter(src_img, target):
    sh, sw, sc = src_img.shape
    th, tw = target[:2]
    if th == sh and tw == sw:
        return src_img.copy()
    tag_img = numpy.zeros((th, tw, sc), dtype=numpy.uint8)
    rh, rw = float(sh) / th, float(sw) / tw  # 原图/目标图宽高比例
    for tc in range(sc):
        for i in range(th):
            for j in range(tw):
                sch = (i + 0.5) * rh - 0.5  # 原图与目标图几何中心重合后的高坐标
                scw = (j + 0.5) * rw - 0.5  # 原图与目标图几何中心重合后的宽坐标

                scw0 = int(numpy.floor(scw))  # 近似宽坐标    无论放大缩小都是向下
                scw1 = min(scw0 + 1, sw - 1)  # 近似宽坐标+1  猜测 如果是放大目标图取sch0+1 缩小目标图应该是取sw-1
                sch0 = int(numpy.floor(sch))  # 近似高坐标
                sch1 = min(sch0 + 1, sh - 1)  # 近似高坐标+1

                # 根据原图两点近似像素获得p0近似像素
                p0 = (scw1 - scw) * src_img[sch0, scw0, tc] + (scw - scw0) * src_img[sch0, scw1, tc]
                # 根据原图两点近似像素获得p1近似像素
                p1 = (scw1 - scw) * src_img[sch1, scw0, tc] + (scw - scw0) * src_img[sch1, scw1, tc]
                tag_img[i, j, tc] = int((sch1 - sch) * p0 + (sch - sch0) * p1)  # 根据p0和p1获得目标图近似像素
    return tag_img


# 3.bgr2gray
def bgr2gary(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    src_img = cv2.imread('lenna.png')

    near_img = near_inter(src_img, (800, 800))
    bilinear_img = bilinear_inter(src_img, (800, 800))
    cv2.imshow('src_img', src_img)
    cv2.imshow('near_img', near_img)
    cv2.imshow('bilinear_img', bilinear_img)
    cv2.waitKey()

    gray_img=bgr2gary(src_img)
    cv2.imshow('src_img', src_img)
    cv2.imshow('gray_img',gray_img)
    cv2.waitKey()