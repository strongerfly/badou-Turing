import numpy as np
import cv2
def nearest_interp(img,dim_out):
    input_w,input_h,c=img.shape
    out_w,out_h=dim_out[0],dim_out[1]
    scale_x,scale_y=out_w/input_w,out_h/input_h
    out_img=np.zeros((out_w,out_h,c),dtype=np.uint8)
    for x in range(out_w):
        for y in range(out_h):
            out_img[x][y]=img[round(x/scale_x)][round(y/scale_y)]

    return out_img
if __name__=="__main__":
    input_img=cv2.imread("lenna.png")
    print("before",input_img.shape)
    cv2.imshow("before", input_img)
    out_img=nearest_interp(input_img,(900,900))
    print("after",out_img.shape)
    cv2.imshow("after",out_img)
    cv2.waitKey(0)






