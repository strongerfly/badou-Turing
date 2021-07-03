import numpy as np
import cv2
#读取图像
img = cv2.imread("D:/GoogleDownload/lenna.png")
inter_img = np.zeros((800, 800, 3), dtype=np.uint8)
s_d_w = float(img.shape[0]) / 800
s_d_h = float(img.shape[1]) / 800
for k in range(3):
    for i in range(800):
        for j in range(800):
            x = ((i+0.5) * s_d_w) - 0.5
            y = ((j+0.5) * s_d_h) - 0.5
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            #if x+1 >= img.shape[0]-1:
                #x = img.shape[0]-2
            #if y+1 >= img.shape[1]-1:
                #y = img.shape[1]-2
            x1 = min(x0 + 1, img.shape[0]-1)
            y1 = min(y0 + 1, img.shape[1] - 1)

            inter_img[i, j, k] = int((y1 -y)*((x1-x )*img[x0, y0, k] + (x -x0)*img[x1, y0, k]) + (y-y0)*((x1-x)*img[x0, y1, k] + (x-x0)*img[x1, y1, k]))

cv2.imshow('inter_img', inter_img)
cv2.waitKey(0)