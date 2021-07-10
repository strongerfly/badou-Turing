import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('D:/GoogleDownload/lenna.png')
img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel1 = np.array([[1, 0, 0], [-1, 0, 0], [0, -1, 1]])
kernel2 = np.array([[-1, -1, 0], [-1, -1, 1], [1, 0, 0]])
kernel3 = np.array([[1, 1, 0], [0, -1, 1], [1, 1, 1]])
kernel = np.array([kernel1, kernel2, kernel3])

# img_out = np.zeros(img.shape, dtype='uint8')
#灰度图进行卷积
def convolu(img, kernel, kernel_size, step):
    width = img.shape[0]
    height = img.shape[1]
    new_w = int((width - kernel_size) / step + 1)
    new_h = int((height - kernel_size) / step + 1)
    img_out = np.zeros([new_w, new_h], dtype='uint8')

    for i in range(new_w):
        for j in range(new_h):
            img_array = img[i:i+3, j:j+3]
            #print('img_gray:', img_array)
            dot_s = np.dot(img_array, kernel)
            #print(dot_s)
            out_s = np.sum(dot_s)
            #print(out_s)
            img_out[i, j] = out_s

    return img_out
# 彩色图卷积
def convolu3(img, kernel, kernel_size, step):
    width = img.shape[0]
    height = img.shape[1]
    new_w = int((width - kernel_size) / step + 1)
    new_h = int((height - kernel_size) / step + 1)
    img_out = np.zeros([new_w, new_h], dtype='uint8')
    out_a = []
    for k in range(3):
        for i in range(new_w):
            for j in range(new_h):
                img_array = img[i:i+3, j:j+3]
                #print('img_gray:', img_array)
                dot_s = np.dot(img_array, kernel)
                #print(dot_s)
                out_s = np.sum(dot_s)
                #print(out_s)
        out_a.append(out_s)
        out_a.sum()
        print(out_a)
        img_out[i, j] = out_a

    return img_out

iii = convolu(img, kernel1, 3, 1)
cv2.imshow("img", iii)
cv2.waitKey(0)