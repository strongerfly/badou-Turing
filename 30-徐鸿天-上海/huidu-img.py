import cv2

import numpy as np

if __name__ == "__main__":
    img_path = "../../bd_ai/30-徐鸿天-上海/1.jpg"
    img = cv2.imread(img_path)
    #获取图片的宽和高
    width,height = img.shape[:2][::-1]

    #将图片转为灰度图
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imshow("img_gray",img_gray)



    print("img_gray shape:{}".format(np.shape(img_gray)))
    cv2.waitKey(0)

