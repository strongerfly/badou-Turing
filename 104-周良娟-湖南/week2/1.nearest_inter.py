import numpy as np
import cv2

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all' #默认为'last'
def nearest(img, large_height, large_width):
    '''
    用最近差值实现图像的上采样
    large_height: 扩大后图像的长
    large_width: 扩大后图像的宽
    '''
    height, width, channels = img.shape
    # 扩大的图像的初始化, 注意数据的类型为np.uint8
    enlarge_image = np.zeros([large_height, large_width, channels], np.uint8)
    # 扩大后图像和原图像的比例
    enlar_h = large_height / height
    enlar_w = large_width / width
    for h1 in range(large_height):
        for w1 in range(large_width):
            h0 = int(h1 / enlar_h)  # 取整实现了公式的定义
            w0 = int(w1 / enlar_w)
            enlarge_image[h1, w1] = img[h0, w0]
    return enlarge_image

if __name__ == '__main__':
    path = '/Users/snszz/PycharmProjects/CV/第二周/代码/lenna.png'
    img = cv2.imread(path)
    enlarge_image = nearest(img, 1000, 800)
    #     print('enlarge_image \n', enlarge_image)
    print('shape:' , enlarge_image.shape)
    cv2.imshow('nearest interpolation',  enlarge_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








