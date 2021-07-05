import numpy as np
import cv2

def nearest(image, target_size):
    if target_size[0] < image.shape[0] or target_size[1] < image.shape[1]:
        raise ValueError("target image must bigger than input image")
    # 1：按照尺寸创建目标图像
    target_image = np.zeros(shape=(*target_size, 3))
    # 2:计算height和width的缩放因子
    alpha_h = target_size[0 ] /image.shape[0]
    alpha_w = target_size[1 ] /image.shape[1]

    for tar_x in range(target_image.shape[0 ] -1):
        for tar_y in range(target_image.shape[1 ] -1):
            # 3:计算目标图像人任一像素点
            # target_image[tar_x,tar_y]需要从原始图像
            # 的哪个确定的像素点image[src_x, xrc_y]取值
            # 也就是计算坐标的映射关系
            src_x = round(tar_x /alpha_h)
            src_y = round(tar_y /alpha_w)

            # 4：对目标图像的任一像素点赋值
            target_image[tar_x, tar_y] = image[src_x, src_y]

    return target_image


if __name__ == '__main__':
    img_path = 'lenna.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_shape = (img.shape[0], img.shape[1])
    dst_shape = (2*img_shape[0], 2*img_shape[1])
    dst_img = nearest(img, dst_shape)
    cv2.imwrite('lala.png', dst_img)