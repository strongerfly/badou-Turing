import cv2
import numpy as np


# match匹配
def drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch):
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]

    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1_gray
    vis[:h2, w1:w1 + w2] = img2_gray

    p1 = [kpp.queryIdx for kpp in goodMatch]  # kp1的序列
    p2 = [kpp.trainIdx for kpp in goodMatch]  # kp2的序列

    post1 = np.int32([kp1[pp].pt for pp in p1])
    post2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(post1, post2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
# cv2.line()用于在图像中划线的函数。第一个img是要画的线所在的图像，第二个参数pt1是直线起点，第三个参数pt2直线终点
# 第四个参数color是直线的颜色，第五个参数thickness是线条的粗细。

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1_gray = cv2.imread("iphone1.png")
img2_gray = cv2.imread("iphone2.png")

# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SURF()

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# BFmatcher with default parms
bf = cv2.BFMatcher(cv2.NORM_L2)
# 暴力匹配（一个一个遍历匹配），有两个参数可选，第一个是normType它用来指定要使用的距离测试类型，默认值为cv2.NORM_L2,适用于SIFT和SURE算法
# 对于二进制描述符的ORB、BRIEF、BRISK算法，则使用cv2.NORM_HAMMING
# 第二个参数是布尔变量crossCheck,默认值为False，如果设置为True匹配条件就会更加严格，只有俩特征点双向的距离都是最近的时候才会返回最几家匹配（i,j）
matches = bf.knnMatch(des1, des2, k=2)  # 利用匹配器匹配两个描述符的相近程度，返回K个最佳匹配项

goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

drawMatchesKnn_cv2(img1_gray, kp1, img2_gray, kp2, goodMatch[:20])

cv2.waitKey(0)
cv2.destroyAllWindows()