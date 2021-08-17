
import cv2
import numpy as np

def bilinear_interp(src_image, dst_h, dst_w):
    src_h, src_w, channels = src_image.shape
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    dst_image = np.zeros([dst_h, dst_w, channels], src_image.dtype)
    if src_h == dst_h and src_w == dst_w:
        return src_image.copy()

    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                x1 = int(np.floor(src_x))   # 向下取整
                x2 = min(x1 + 1, src_w - 1)
                y1 = int(np.floor(src_y))
                y2 = min(y1 + 1, src_h - 1)

                f1 = (x2 - src_x) * src_image[y1, x1, i] + (src_x - x1) * src_image[y1, x2, i]
                f2 = (x2 - src_x) * src_image[y2, x1, i] + (src_x - x1) * src_image[y2, x2, i]
                dst_image[dst_y, dst_x, i] = int((y2 - src_y) * f1 + (src_y - y1) * f2)

    return dst_image

if __name__ == '__main__':
    # 读入图像数据
    img = cv2.imread('lenna.png')
    cv2.imshow("lenna image", img)

    # 双线性插值
    dstImg = bilinear_interp(img, 700, 700)
    print(dstImg)
    cv2.imshow("bilinear interpolation", dstImg)
    cv2.waitKey()
    cv2.destroyAllWindows()