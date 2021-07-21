import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[315, 39], [416, 50], [236, 476], [407, 477]])
dst = np.float32([[0, 0], [140, 0], [0, 488], [140, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (140, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
