import cv2
import numpy as np

def function(img):
    h,w,channel=img.shape
    dst_h,dst_w=700,700
    dst=np.zeros((dst_h,dst_w,channel),np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            x=(i+0.5)/dst_h*h-0.5
            y=(j+0.5)/dst_w*w-0.5
            # 向下取整
            x_int=int(x)
            y_int=int(y)

            u=x-x_int
            v=y-y_int

            # 防止下标溢出
            if x_int==h-1 or  y_int==w-1:
                x_int=x_int-1
                y_int=y_int-1

            # 套用公式
            dst[i][j]=(1-u)*(1-v)*img[x_int][y_int]+(1-u)*v*img[x_int][y_int+1]+u*(1-v)*img[x_int+1][y_int]+u*v*img[x_int+1][y_int+1]
    return dst
if __name__ == "__main__":
    img = cv2.imread('./lenna.png')
    bilinear = function(img)
    cv2.imshow('old', img)
    cv2.imshow('bilinear', bilinear)
    cv2.waitKey()