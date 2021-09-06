import numpy as np
import cv2

def BGRToGray(image):   # 灰度化函数
    height, width = image.shape[:2]
    image_gray = np.zeros([height,width], image.dtype)
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            image_gray[i, j] = pixel[2]*0.11 + pixel[1]*0.59 + pixel[0]*0.3

    return image_gray

def GrayToBinary(image):    # 二值化函数
    rows, columns = image.shape
    image_Bi = np.zeros([rows, columns], image.dtype)
    for i in range(rows):
        for j in range(columns):
            if image[i, j] <= 127:
                image_Bi[i, j] = 0
            else:
                image_Bi[i, j] = 255
    return image_Bi

if __name__ == '__main__':
    img = cv2.imread("lenna.png")   # opencv读进来的图片通道排列是BGR
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 由BGR通道排列转为主流的RGB排列通道

    # 灰度化
    img_gray = BGRToGray(img)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("image pixel value: \n", img_gray)
    cv2.imshow("gray image",img_gray)

    # 二值化
    image_bi = GrayToBinary(img_gray)
    print("binary image value: \n", image_bi)
    cv2.imshow("binary image", image_bi)
    cv2.waitKey()
    cv2.destroyWindow("gray image")
    cv2.destroyWindow("binary image")