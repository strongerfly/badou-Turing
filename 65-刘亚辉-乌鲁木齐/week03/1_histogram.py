import numpy as np
import cv2
import matplotlib.pyplot as plt


def calcu_histogram(image, src_rows, src_cols):
    # 统计原图中每个像素出现的个数，即直方图
    histogram = {}
    for src_row in range(src_rows):
        for src_col in range(src_cols):
            pixel = image[src_row, src_col]
            if pixel not in histogram:
                histogram[pixel] = 1
            else:
                histogram[pixel] += 1
    return histogram


def hist_equal(image, dst_rows, dst_cols):
    src_rows, src_cols = image.shape
    # 构造新图
    equal_image = np.zeros((dst_rows, dst_cols), dtype=np.uint8)
    # 统计原图中每个像素出现的个数，即直方图
    histogram = calcu_histogram(image, src_rows, src_cols)

    # 通过直方图对原像素值进行均衡化，得到新图像素值
    hist_sum = 0
    equal_pixel = {}
    for pixel in sorted(list(histogram)):   # 循环前先对原像素值排序
        # 累加直方图
        hist_sum += histogram[pixel]
        # 根据公式计算新图像素值
        equal_pixel[pixel] = (256*hist_sum/(dst_rows*dst_cols)) - 1

    # 根据像素对应关系更新新图像素值
    for dst_row in range(dst_rows):
        for dst_col in range(dst_cols):
            pixel = image[dst_row, dst_col]
            equal_image[dst_row, dst_col] = equal_pixel[pixel]

    return equal_image


def hist_equal_colored(image, rows, cols, channels):
    channel_list = []
    for chn in range(channels):
        # 对每个通道分别进行均衡化处理
        chn_image = image[:, :, chn]
        channel_image = hist_equal(chn_image, rows, cols)
        channel_list.append(channel_image)
    # 合并三个通道
    full_image = np.dstack(channel_list)
    return full_image


image_file = '1.1_lenna_grey_hand.png'
image = plt.imread(image_file)

# 灰度图
plt.subplot(121).set_title('Gray')
plt.imshow(image ,cmap='gray')

# 直方图均衡化后的图像
dst_rows, dst_cols = 512, 512
equal_image = hist_equal(image, dst_rows, dst_cols)
plt.subplot(122).set_title('Histogram Equalization')
plt.imshow(equal_image, cmap='gray')

plt.show()

# 原图像素直方图
plt.subplot(311).set_title('Original')
plt.hist(image.ravel(), 256)

# 均衡化后的像素直方图(手动)
plt.subplot(312).set_title('Equalization(by hand)')
plt.hist(equal_image.ravel(), 256)

# 均衡化后的像素直方图(调包)
cv_image = cv2.imread(image_file)
image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
cv_equalize = cv2.equalizeHist(image_gray)
plt.subplot(313).set_title('Equalization(by cv2)')
plt.hist(cv_equalize.ravel(), 256)

plt.tight_layout()
plt.show()

# RGB三通道图像
rgb_image = plt.imread('lenna.png')
plt.subplot(121).set_title('RGB Original')
plt.imshow(rgb_image)

# 均衡化后的像素直方图(手动)
rows, cols, channels = rgb_image.shape
rgb_equalized = hist_equal_colored(rgb_image, dst_rows, dst_cols, channels)
plt.subplot(122).set_title('RGB Equalization(by hand)')
plt.imshow(rgb_equalized)

plt.show()
