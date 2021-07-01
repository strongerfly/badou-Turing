import cv2
import numpy as np
import math

def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def BiLinear_interpolation(img,dstH,dstW):

    scrH, scrW, _ = img.shape
    emptyImage = np.zeros((dstH,dstW,3,), np.uint8)
    for k in range(3):
        for i in range(dstH):
            for j in range(dstW):
                # 首先找到在原图中对应的点的(X, Y)坐标
                src_x = (j+0.5)*(scrW/dstW)-0.5
                src_y = (i+0.5)*(scrH/dstH)-0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, scrW - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, scrH - 1)
                # 计算插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, k] + (src_x - src_x0) * img[src_y0, src_x1, k]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, k] + (src_x - src_x0) * img[src_y1, src_x1, k]

                emptyImage[i, j, k] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return emptyImage


if __name__ == '__main__':
    src_img = cv2.imread("lena.jpg")

    BiLinear_img_1 = BiLinear_interpolation(src_img,600,600)
    cv_show(BiLinear_img_1, "tar_img")

    BiLinear_img_2 = cv2.resize(src_img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv_show(BiLinear_img_2, "tar_img")