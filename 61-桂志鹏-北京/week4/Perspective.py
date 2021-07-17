# 3.透视变换实现。

import cv2
import numpy as np

img = cv2.imread('img/photo1.jpg')
print(img.shape)
img_copy = img.copy()  # 复制一份图像做转换

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

wrap_matrix = cv2.getPerspectiveTransform(src, dst)
print('wrap_matrix: ', wrap_matrix)

dst_img = cv2.warpPerspective(img_copy, wrap_matrix, (337, 490))

cv2.imshow('img', img)
cv2.imshow('matrix_img', dst_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
