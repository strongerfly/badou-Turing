import numpy as np
import cv2

def imgGray_Lum(im):
    imgarray = np.array(im, dtype=np.float32)  # 以浮点型读取图像数据
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(rows):
        for j in range(cols):
            imgarray[i, j, :] = np.clip((imgarray[i, j, 0] + imgarray[i, j, 1] + imgarray[i, j, 2]) * 0.3333, 0.0,
                                        255.0)
    return imgarray.astype(np.uint8)


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = imgGray_Lum(img)
    cv2.imshow('gray', dst)
    cv2.waitKey()