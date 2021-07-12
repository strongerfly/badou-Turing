import cv2
import numpy as np


def conv_func():
    img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
    print(img)

    #define 3*3 convolution kenel
    k = np.array([[-1,-1,-1],
                  [-1,9,-1],
                  [-1,-1,-1]])
    w,h=img.shape
    conv_res = []
    for i in range(w-3):
        temp = []
        for j in range(h-3):
            a = img[i:i+3,j:j+3]
            temp.append(np.sum(np.multiply(a,k)))
        conv_res.append(temp)
    result = np.array(conv_res)
    # result = (result-np.min(result))*255/(np.max(result)-np.min(result))
    result = (result*255).astype(np.uint8)
    cv2.imshow("img", img)
    cv2.imshow("conv",result)
    cv2.waitKey(0)

if __name__=="__main__":
    conv_func()