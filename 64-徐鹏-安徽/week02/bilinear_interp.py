'''
关于img.shape[0]、[1]、[2]
img.shape[0]：图像的垂直尺寸（高度）
img.shape[1]：图像的水平尺寸（宽度）
img.shape[2]：图像的通道数

在矩阵中，[0]就表示行数，[1]则表示列数。
'''
import numpy as np
import cv2

def bilinear_interpolation(img,img_size):
    '''
    双线性差值
    :param img: 原始图像
    :param img_size: 需要插值的图像大小
    :return: 插值后的图像
    '''
    origin_img_h,origin_img_w,img_channals=img.shape
    target_img_h,target_img_w=img_size[1],img_size[0]
    print("origin_img_h, origin_img_w = ", origin_img_h, origin_img_w)
    print("target_img_h, target_img_w = ", target_img_h, target_img_w)
    #大小相同，直接复制
    if origin_img_h == target_img_h and origin_img_w == target_img_w:
        return img.copy()
    target_img=np.zeros((target_img_h,target_img_w,3),dtype=np.uint8)
    #长、宽缩放比例
    scale_x,scale_y=float(origin_img_w)/target_img_w,float(origin_img_h)/target_img_h
    for i in range(3):
        for target_y in range(target_img_h):
            for target_x in range(target_img_w):
                #两个图像的几何中心重合，通过目标像素点的坐标找到原图像像素点的坐标
                origin_x=(target_x+0.5)*scale_x-0.5
                origin_y=(target_y+0.5)*scale_y-0.5
                print('origin_x:',type(origin_x))#origin_x: <class 'float'>
                #找到用于双线性插值的点的坐标
                origin_x0=int(np.floor(origin_x))
                origin_x1=min(origin_x0+1,origin_img_w-1)
                origin_y0=int(np.floor(origin_y))
                origin_y1=min(origin_y0+1,origin_img_h-1)

                #计算插值
                interp1=(origin_x1-origin_x)*img[origin_y0,origin_x0,i]+(origin_x-origin_x0)*img[origin_y0,origin_x1,i]
                interp2=(origin_x1-origin_x)*img[origin_y1,origin_x0,i]+(origin_x-origin_x0)*img[origin_y1,origin_x1,i]
                target_img[target_y,target_x,i]=int((origin_y1-origin_y)*interp1+(origin_y-origin_y0)*interp2)

    return target_img

if __name__ == '__main__':
     img=cv2.imread('lenna.png')
     target_img=bilinear_interpolation(img,(700,700))
     cv2.imshow('target_img',target_img)
     cv2.waitKey()
     cv2.destroyAllWindows()




















































