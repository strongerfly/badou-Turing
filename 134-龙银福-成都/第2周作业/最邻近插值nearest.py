import cv2
import numpy as np

def nearest_interp(image, h, w):
    height, width, channels = image.shape
    emptyimage = np.zeros([h, w, channels], np.uint8)
    sh = h / height
    sw = w / width
    for i in range(h):
        for j in range(w):
            x = int(i / sh)
            y = int(j / sw)
            emptyimage[i, j] = image[x, y]

    return emptyimage

if __name__ == '__main__':
    image = cv2.imread('lenna.png')
    print("---- lenna image ----")
    print(image)
    cv2.imshow("lenna image", image)

    image_nearest = nearest_interp(image, 400, 400)
    print("---- 最邻近插值 ----")
    print(image_nearest)
    print("原始图像尺寸大小：", image.shape)
    print("插值后的图像大小：", image_nearest.shape)
    cv2.imshow("nearest interpolation", image_nearest)
    cv2.waitKey()
    cv2.destroyAllWindows()