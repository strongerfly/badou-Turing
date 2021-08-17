# -*- coding: utf-8 -*-
 
import numpy as np
import cv2

def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # if(dst_y == dst_x == 0):
                #     print(src_x0,src_x1,src_y0,src_y1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img

def my_bilinear_interpolation(img,out_dim):
    h, w, c = img.shape
    nh, nw = out_dim
    fx, fy = w/nw , h/nh

    x_index, y_index = np.meshgrid(np.arange(0, nw), np.arange(0, nh))
    src_x = (x_index + 0.5) * fx - 0.5
    src_y = (y_index + 0.5) * fy - 0.5

    src_x0 = np.floor(src_x).astype(np.int) #np.floor为了跟老师结果一致
    src_x1 = np.clip(src_x0 + 1, 0, w - 1)
    src_y0 = np.floor(src_y).astype(np.int)
    src_y1 = np.clip(src_y0 + 1, 0, h - 1)
    #print(src_x0[0,0],src_x1[0,0],src_y0[0,0],src_y1[0,0])
    temp0 = (src_x1 - src_x)[...,np.newaxis] * img[src_y0,src_x0,:] + (src_x - src_x0)[...,np.newaxis] * img[src_y0,src_x1,:]
    temp1 = (src_x1 - src_x)[...,np.newaxis] * img[src_y1,src_x0,:] + (src_x - src_x0)[...,np.newaxis] * img[src_y1,src_x1,:]
    dst_img = ((src_y1 - src_y)[...,np.newaxis] * temp0 + (src_y - src_y0)[...,np.newaxis] * temp1).astype(np.uint8)
    return dst_img

if __name__ == '__main__':
    import time

    img = cv2.imread('../images/lenna.png')
    t1 = time.time()
    dst = bilinear_interpolation(img,(700,700))
    t2 = time.time()
    dst2 = my_bilinear_interpolation(img,(700,700))
    t3 = time.time()

    error = np.sum(dst2 - dst)
    info = "逐像素循环计算耗时%.3fs，numpy api运算耗时%.3fs,两方式总像素误差%.2f" % (t2 - t1, t3 - t2, error)
    print(info)
    cv2.imshow('bilinear interp', np.concatenate((dst, dst2), axis=1))
    cv2.waitKey()
