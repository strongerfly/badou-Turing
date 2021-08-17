import numpy as np
import cv2
def bilinear_interp(input_img,out_dim):
    input_w,input_h,channel=input_img.shape
    out_w,out_h=out_dim[0],out_dim[1]
    if(input_w==out_w and input_h==out_h):
        return input_img
    out_img=np.zeros((out_w,out_h,channel),dtype=np.uint8)
    scale_x,scale_y=float(input_w )/out_w,float(input_h)/out_h
    for out_x in range(out_w):
        for out_y in range(out_h):
            for i in range(3):
                #中心对齐
                input_x=(out_x+0.5)*scale_x-0.5
                input_y=(out_y+0.5)*scale_y-0.5
                #原图四个角点
                input_x0=int(input_x)
                input_x1=min(input_x0+1,input_w-1)
                input_y0=int(input_y)
                input_y1=min(input_y0+1,input_h-1)
                #计算插值
                temp0 = (input_x1 - input_x) * input_img[input_x0, input_y0, i] + (input_x - input_x0) * input_img[
                    input_x1, input_y0, i]
                temp1 = (input_y1 - input_y) * input_img[input_x0, input_y1, i] + (input_y - input_y0) * input_img[
                    input_x1, input_y1, i]
                out_img[out_x, out_y, i] = int((input_y1 - input_y) * temp0 + (input_y - input_y0) * temp1)

    return out_img
if __name__=="__main__":
    input_img=cv2.imread("lenna.png")
    print("imput_img",input_img.shape)
    cv2.imshow("imput_img",input_img)
    out_img=bilinear_interp(input_img,(900,900))
    print("out_img",out_img.shape)
    cv2.imshow("out_img",out_img)
    cv2.waitKey(0)


