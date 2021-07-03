from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

# 最近邻插值算法

def NN_interpolation(img, dstH, dstW):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))  #对浮点数进行近似取值，保留几位小数,第二个参数不写，默认为整数
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg

im_path = 'lenna.png'
image =np.array(Image.open(im_path))   #image.open默认读取彩色图片顺序是RGB,而从v.imread读取为BGR

image1 = NN_interpolation(image, image.shape[0] * 2, image.shape[1] * 2)
image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')

image1.save('out1.png')
img = cv2.imread("out1.png")
cv2.imshow("image show ",img)
cv2.waitKey()
