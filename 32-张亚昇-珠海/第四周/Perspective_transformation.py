import cv2
import numpy as np

img = cv2.imread("./photo1.jpg")

result = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
#使用opencv中的算法计算透视变换矩阵
pt = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(pt)
#使用opencv中算法进行透视变换
result1 = cv2.warpPerspective(img, pt, (400, 500))
cv2.imshow("src", img)
cv2.imshow("dst", result1)
cv2.waitKey(0)