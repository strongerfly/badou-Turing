import numpy as np
import cv2
def rgb2gray(img):
    w,h,c=img.shape
    gray_img=np.zeros((w,h),dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            gray_img[x][y]=0.3*img[x][y][0]+0.59*img[x][y][1]+0.11*img[x][y][2]
    return gray_img
if __name__=="__main__":
    img=cv2.imread("lenna.png")
    cv2.imshow("before", img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print("before",img.shape)
    gray_img=rgb2gray(img)
    print("after", gray_img.shape)
    cv2.imshow("after", gray_img)
    cv2.waitKey(0)

