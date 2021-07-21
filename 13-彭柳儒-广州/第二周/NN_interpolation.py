import cv2
import numpy as np
import math

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()



def NN_interpolation(src_img,dst_h,dst_w):
    src_h, src_w,_ = src_img.shape
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            src_y = round((i+1)*(src_h/dst_h))
            src_x = round((j+1)*(src_w/dst_w))

            dst_img[i,j] = src_img[src_y-1,src_x-1]
    return dst_img


if __name__ == '__main__':
    src_img = cv2.imread("lena.jpg")
    NN_img = NN_interpolation(src_img,512,512)
    cv_show(NN_img, "NN_img")