import cv2
import numpy as np

def gauss_noise(img, sigma, mu):
    '''
    :param img: 灰度图
    :param sigma:
    :param mu:
    :return: 带噪声的图像
    '''
    h, w = img.shape
    noise = sigma * np.random.randn(h, w) + mu
    new_img = img + noise
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    return  new_img.astype(np.uint8)


if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path, 0)  # gray image
    cv2.imshow('lenna', img)
    cv2.waitKey(0)
    new_img = gauss_noise(img, 20, 10)
    cv2.imshow('lenna noise', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




