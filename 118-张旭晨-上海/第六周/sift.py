# -*- coding: utf-8 -*-
# 导入相应的包
import cv2
import numpy as np


img = cv2.imread('lenna.jpg')
img2 = cv2.imread('lenna1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 1、构建尺度空间
sift = cv2.SIFT_create()
# 2、探测关键点并生成关键点描述
keypoints1, descriptor1 = sift.detectAndCompute(gray, None)
keypoints2, descriptor2= sift.detectAndCompute(gray2, None)
# 构显出关键点点信息
img_sift = cv2.drawKeypoints(image=gray, outImage=gray,
                              keypoints=keypoints1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                              color=(100, 200, 255))
img_sift2 = cv2.drawKeypoints(image=gray2, outImage=gray2,
                              keypoints=keypoints2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                              color=(100, 200, 255))
# 3、构建匹配器，采用L2范数归一化
bf = cv2.BFMatcher_create(normType=cv2.NORM_L2)
# 4、匹配两图关键点描述情况，最多允许在train图上有两个点匹配
macthes = bf.knnMatch(descriptor1, descriptor2, k=2)
good_match = []
# 如果第一个点的欧氏距离是第二个点的一半或更少，则为最佳关键点
for m, n in macthes:
    if m.distance < 0.50 * n.distance:
        good_match.append(m)
# 构建比对图
h1, w1 = img.shape[:2]
h2, w2 = img2.shape[:2]
img_match = np.zeros((max(h1, h2), w1 + w2, 3), img.dtype)
# 两张图放在一张图上
img_match[:h1, :w1] = img
img_match[:h2, w1:] = img2
# 锚定两张图的关键点
qi = [kpp.queryIdx for kpp in good_match]
ti = [kpp.trainIdx for kpp in good_match]
p1 = np.int32([keypoints1[p].pt for p in qi])
p2 = np.int32([keypoints2[p].pt for p in ti])+(w1, 0)
# 将两张图匹配关键点进行连线
for (x1, y1), (x2, y2) in zip(p1, p2):
    cv2.line(img_match, (x1, y1), (x2, y2), (100, 200, 255))
cv2.namedWindow("match",cv2.WINDOW_NORMAL)
cv2.imshow("match", img_match)
# cv2.imshow('sift_keypoints', img_sift)
# cv2.imshow('sift_keypoints2', img_sift2)
cv2.waitKey(0)
cv2.destroyAllWindows()
