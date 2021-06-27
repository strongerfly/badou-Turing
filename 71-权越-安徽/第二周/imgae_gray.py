import  cv2
import  numpy as np

def function(img):
    h,w=img.shape[:2]
    img_dst=np.zeros((h,w),img.dtype)
    for i in range (h):
        for j in range (w):
            m=img[i][j]
            img_dst[i][j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
    return  img_dst
if __name__ == "__main__":
    img = cv2.imread('./lenna.png')
    gray = function(img)
    cv2.imshow('old', img)
    cv2.imshow('gray', gray)
    cv2.waitKey()