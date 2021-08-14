
import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread("lenna.png",1)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#canny
canny_img=cv2.Canny(img_gray,100,300)
#sobel
sobelx_img=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)
sobely_img=cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=3)

#laplace
laplace_img=cv2.Laplacian(img_gray,cv2.CV_64F,ksize=5)
plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
plt.subplot(232), plt.imshow(canny_img,"gray"), plt.title("Canny")
plt.subplot(233), plt.imshow(sobelx_img, "gray"), plt.title("Sobel_x")
plt.subplot(234), plt.imshow(sobely_img, "gray"), plt.title("Sobel_y")
plt.subplot(235), plt.imshow(laplace_img,  "gray"), plt.title("Laplace")

plt.show()