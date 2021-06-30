
import cv2
import numpy as np

def my_resize_nearst(img,fx,fy):
    h,w,c =img.shape
    nh,nw = int(h*fy),int(w*fx)
    x_index,y_index = np.meshgrid(np.arange(0,nw),np.arange(0,nh))
    xy_index = np.concatenate((x_index[...,np.newaxis],y_index[...,np.newaxis]),axis=2).astype(np.float32)
    #print(xy_index.shape,xy_index[10,20])
    factors = np.array([1/fx,1/fy])
    xy_index = (xy_index*factors).astype(np.int) #映射坐标

    x_src_pos = xy_index[:, :, 0]
    y_src_pos = xy_index[:, :, 1]

    resizeImage = img[y_src_pos,x_src_pos,:]
    return resizeImage

if __name__=="__main__":
    import time
    img=cv2.imread("lenna.png")
    h,w,c=img.shape
    fx = 800 / w
    fy = 800 / h

    t1 = time.time()
    # 循环计算
    resize_img_1 = np.zeros((800, 800, c), np.uint8)
    for i in range(800):
        for j in range(800):
            x = int(i / fy)
            y = int(j / fx)
            resize_img_1[i, j] = img[x, y]
    t2 = time.time()
    #numpy api 实现
    resize_img_2 = my_resize_nearst(img,fx,fy)
    t3 = time.time()
    error = np.sum(resize_img_1-resize_img_2)
    info = "逐像素循环计算耗时%.3fs，numpy api运算耗时%.3fs,两方式总像素误差%.2f" % (t2 - t1, t3 - t2,error)
    print(info)
    cv2.imshow('src', img)
    cv2.imshow('result', np.concatenate((resize_img_1,resize_img_2),axis=1))
    cv2.waitKey(0)


