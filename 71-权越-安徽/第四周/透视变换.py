import cv2
import numpy as np

img = cv2.imread('lenna.png')

result3 = img.copy()

'''
注意 这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[228, 215], [723, 215], [228, 661], [723, 661]])
dst = np.float32([[91, 124], [856, 124], [91, 887], [856, 887]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (800, 600))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
