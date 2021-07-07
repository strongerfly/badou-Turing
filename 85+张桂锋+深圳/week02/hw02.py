import cv2
import numpy as np
'''
第二周作业：
1）最邻近插值实现
2）双线性插值实现
3）rgb2gray
'''
# 最邻近插值实现
def nearest_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    print ("channel = ", channel)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    scale_h = dst_h/src_h
    scale_w = dst_w/src_w
    emptyImage=np.zeros((dst_h,dst_w,channel),np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i/scale_h)
            y = int(j/scale_w)
            emptyImage[i,j] = img[x,y]
    return emptyImage

# 双线性插值实现
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    print ("channel = ", channel)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    scale_y = float(src_h)/dst_h
    scale_x = float(src_w)/dst_w
    emptyImage=np.zeros((dst_h,dst_w,channel),np.uint8)
    for i in range(channel) :
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                #找到对应的4个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                tmp0 = (src_x1 - src_x)*img[src_y0,src_x0,i] +(src_x - src_x0)*img[src_y0,src_x1,i]
                tmp1 = (src_x1 - src_x)*img[src_y1,src_x0,i] +(src_x - src_x0)*img[src_y1,src_x1,i]
                emptyImage[dst_y,dst_x,i]= int((src_y1 - src_y)* tmp0 + (src_y - src_y0)*tmp1)
    return emptyImage

# rgb2gray
def rgb2gray(img):
    src_h, src_w, channel = img.shape
    print ("src_h, src_w = ", src_h, src_w)
    print ("channel = ", channel)
    emptyImage = np.zeros((src_h, src_w, channel), np.uint8)
    for i in range(src_h):
        for j in range(src_w):
            m = img[i,j] #取出当前high和wide中的BGR坐标
            emptyImage[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
    return emptyImage


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img1 = nearest_interpolation(img,(800,800))
    img2 = bilinear_interpolation(img, (800, 800))
    img3 = rgb2gray(img)
    cv2.imshow("image", img)
    cv2.imshow("nearest_interpolation", img1)
    cv2.imshow("bilinear_interpolation", img2)
    cv2.imshow("rgb2gray", img3)
    cv2.waitKey(0)