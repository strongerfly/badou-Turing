import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#方法1 resize opencv操作
img = cv2.imread("lenna.png")
# img = cv2.imread("C:/Users/ZhongXH2/Desktop/zuoye/cat.jpg")
cv_show("image",img)
# print(img.shape)

# img_new =cv2.resize(img,(0,0),fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
# cv_show("image",img_new)

#方法2 像素操作

def bilinear_interpplation(img,width,height):
    image = np.copy(img)
    src_h,src_w, c = image.shape
    img_new = np.zeros((height,width,3), np.uint8)
    scale_x, scale_y = float(src_w) / width, float(src_h) / height

    for kk in range(3):
        for dst_ii in range(height):
            for dst_jj in range(width):

                #映射到原图位置,srcx srcy为初始点,并且中心对齐（0.5的作用）
                tempx = scale_y * (dst_ii + 0.5) - 0.5
                tempy = scale_x * (dst_jj + 0.5) - 0.5

                #中心不对齐，看右下角边界
                # tempx =  scale_y * dst_ii
                # tempy =  scale_x * dst_jj

                # 输出图片中坐标 （ii，jj）对应至输入图片中的最近的四个点  左上角（x1，y1） 左下角（x2, y2），右上角（x3， y3），右下角(x4，y4)的均值

                #第一个点 左上角 f(i,j)
                x1 = int(np.floor(tempx))
                y1 = int(np.floor(tempy))

                #第二个点 左下角 f(i,j+1)
                x2 = x1
                y2 = y1 + 1

                #右上角 f(i+1,j+1)
                x3 = x1 + 1
                y3 = y1

                #右下角 f(i+1,j)
                x4 = x1 + 1
                y4 = y1 + 1

                # 计算基于初始点的差值
                u = tempx - x1
                v = tempy - y1

                if x4 >= src_h:
                    break
                if y4 >= src_w:
                    break
                # 公式f(i+u,j+v) = (1-u)(1-v)f(i,j) + (1-u)vf(i,j+1) + u(1-v)f(i+1,j) + uvf(i+1,j+1)
                img_new[dst_ii,dst_jj,kk] = (1-u)*(1-v) *int(image[x1,y1,kk])+(1-u)*v*int(image[x2,y2,kk]) + u*(1-v)*int(image[x3,y3,kk]) + u*v*int(image[x4,y4,kk])

    return  img_new

h,w,c = img.shape
# print(img.shape)
img_share = bilinear_interpplation(img,800,600)
cv_show("image",img_share)