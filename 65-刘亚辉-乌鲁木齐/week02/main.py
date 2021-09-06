import matplotlib.pyplot as plt
from trans_funcs import trans_grey_byhand, trans_grey_skimage, trans_binary, nerp, blerp


if __name__ == '__main__':
    # 原图像
    image_file = 'lenna.png'
    image = plt.imread(image_file)
    plt.subplot(231).set_title('original image(512*512)')
    plt.imshow(image)
    # 新图像分辨率
    dst_x, dst_y = 800, 800
    # 1.RGB图像转灰度图像
    grey_image_byhand = trans_grey_byhand(image)
    plt.subplot(232).set_title('grey image(by hand)')
    plt.imshow(grey_image_byhand, cmap='gray')
    grey_image_skimage = trans_grey_skimage(image)
    plt.subplot(233).set_title('grey image(package)')
    plt.imshow(grey_image_skimage, cmap='gray')
    # 2.RGB图像转黑白二值图像
    binary_image = trans_binary(image)
    plt.subplot(234).set_title('binary image')
    plt.imshow(binary_image, cmap='gray')
    # 3.最近邻算法
    nerp_image = nerp(image, dst_x, dst_y)
    plt.subplot(235).set_title('Nearest Interpolation(800*800)')
    plt.imshow(nerp_image)
    # 4.双线性插值算法
    blerp_image = blerp(image, dst_x,dst_y)
    plt.subplot(236).set_title('Linear Interpolation(800*800)')
    plt.imshow(blerp_image)
    plt.show()
