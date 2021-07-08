import cv2
import numpy as np
import math

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def rgb2gray(img):
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            # cv2中读取图片通道顺序为BGR
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    return img_gray

if __name__ == '__main__':
    src_img = cv2.imread("lena.jpg")
    img_gray_1 = cv2.imread("lena.jpg",0)

    img_gray_2 = cv2.cvtColor(src_img,cv2.COLOR_RGB2GRAY)

    img_gray_3 = rgb2gray(src_img)
    cv_show(img_gray_1, "img_gray_1")
    cv_show(img_gray_2, "img_gray_2")
    cv_show(img_gray_3, "img_gray_3")