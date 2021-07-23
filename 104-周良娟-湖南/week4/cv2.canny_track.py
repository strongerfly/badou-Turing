import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray, (3,3), 0)  # gauss filter
    detected_edges = cv2.Canny(detected_edges, lowThreshold,
                               lowThreshold * ratio,
                               apertureSize = kernel_size)
    # just add some colours to edges from original image
    # mask是灰色图，可以用cv2.cvtColor(detected_edges, cv2.COLOR_GRAY2BGR)
    # cv2.bitwise_and()是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
    # 利用掩膜（mask）进行“与”操作，即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除，其余按位操作原理类似只是效果不同而已。
    dst = cv2.bitwise_and(img, img, mask = detected_edges)   # #用原始颜色添加到检测的边缘上,
    cv2.imshow('canny demo', dst)

if __name__ == '__main__':
    lowThreshold = 0
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3

    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('canny demo')

    # 设置调节杆
    #设置调节杠,
    '''
    下面是第二个函数，cv2.createTrackbar()
    共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
    第一个参数，是这个trackbar对象的名字
    第二个参数，是这个trackbar对象所在面板的名字
    第三个参数，是这个trackbar的默认值,也是调节的对象
    第四个参数，是这个trackbar上调节的范围(0~count)
    第五个参数，是调节trackbar时调用(的回调函数名
    '''
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

    CannyThreshold(0)   # initialization
    if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
        cv2.destroyAllWindows()



