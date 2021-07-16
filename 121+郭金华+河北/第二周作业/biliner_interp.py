import numpy as np
import cv2


def fun_biliner_interp(img,shape):
    sur_h,sur_w,sur_ch=img.shape
    dest_h,dest_w=shape[0],shape[1]
    '''判断给定图像与目标图像的大小关系'''
    if sur_h ==dest_h and sur_w ==dest_w:
        return img
    dest_img=np.zeros((dest_h,dest_w,3),dtype=np.uint8)
    deta_x,deta_y=float(sur_w)/dest_w,float(sur_h)/dest_h
    for i in range(3):
        for dst_y in range(dest_h):
            for dst_x in range(dest_w):
                src_x = (dst_x+0.5)*deta_x-0.5
                src_y=(dst_y+0.5)*deta_y-0.5
                '''找到相邻的像素点'''
                src_x0=int(np.floor(src_x))
                src_x1=min(src_x0+1,sur_w-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,sur_h-1)

                temp_0=(src_x1-src_x)*img[src_y0,src_x0,i]+(src_x-src_x0)*img[src_y0,src_x1,i]
                temp_1=(src_x1-src_x)*img[src_y1,src_x0,i]+(src_x-src_x0)*img[src_y1,src_x1,i]
                dest_img[dst_y,dst_x,i]=int((src_y1-src_y)*temp_0+(src_y-src_y0)*temp_1)
    return dest_img


img=cv2.imread('lenna.png')
inter_img=fun_biliner_interp(img,(500,500))

cv2.imshow('interp_img',inter_img)
cv2.imshow('source_img',img)
cv2.waitKey()
cv2.destroyAllWindows()