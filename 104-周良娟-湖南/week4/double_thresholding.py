import cv2
import numpy as np
import matplotlib.pyplot as plt


def double_threshold_check(img_nms, min_thre=None, max_thre=None):
    '''
    :param img:
    :param min_value:
    :param max_value:
    :return:
    '''
    h, w = img_nms.shape
    DT = np.zeros([h, w])
    if not min_thre:
        min_thre = 0.5 * img_nms.mean()
    if not max_thre:
        max_thre = 3 * min_thre
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img_nms[i, j] <= min_thre:
                DT[i, j] = 0
            elif img_nms[i, j] >= max_thre:
                DT[i, j] = 255
            elif (img_nms[i - 1, [j - 1, j, j + 1]] > max_thre).any() or \
                    (img_nms[i + 1, [j - 1, j, j + 1]] > max_thre).any() \
                    or (img_nms[i, [j - 1, j + 1]] > max_thre).any():
                DT[i, j] = 255
    return DT.astype(np.uint8)

def double_threshold_check2(img_nms, min_thre=None, max_thre=None):
    h, w = img_nms.shape
    zhan = []
    if not min_thre:
        min_thre = 0.5 * img_nms.mean()
    if not max_thre:
        max_thre = 3 * min_thre
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img_nms[i, j] <= min_thre:
                img_nms[i, j] = 0
            elif img_nms[i, j] >= max_thre:
                img_nms[i, j] = 255
                zhan.append([i, j])  # 存的是高阈值的索引

    while not len(zhan) == 0:
        id1, id2 = zhan.pop()   # 出栈
        tmp = img_nms[id1-1:id1+2, id2-1:id2+2]
        if (min_thre < tmp[0, 0] < max_thre):
            img_nms[id1 - 1, id2 - 1] = 255    # 标记为边缘
            zhan.append([id1 - 1, id2 - 1])   # 入栈
        if (min_thre < tmp[0, 1] < max_thre):
            img_nms[id1 - 1, id2] = 255
            zhan.append([id1 - 1, id2])
        if (min_thre < tmp[0, 2] < max_thre):
            img_nms[id1 - 1, id2 + 1] = 255
            zhan.append([id1 - 1, id2 + 1])
        if (min_thre < tmp[1, 0] < max_thre):
            img_nms[id1, id2 - 1] = 255
            zhan.append([id1, id2 - 1])
        if (min_thre < tmp[1, 2] < max_thre):
            img_nms[id1, id2 + 1] = 255
            zhan.append([id1, id2 + 1])
        if (min_thre < tmp[2, 0] < max_thre):
            img_nms[id1 + 1, id2 - 1] = 255
            zhan.append([id1 + 1, id2 - 1])
        if (min_thre < tmp[2, 1] < max_thre):
            img_nms[id1 + 1, id2] = 255
            zhan.append([id1 + 1, id2])
        if (min_thre < tmp[2, 2] < max_thre):
            img_nms[id1 + 1, id2 + 1] = 255
            zhan.append([id1 +1, id2 + 1])

    for i in range(h):
        for j in range(w):
            if img_nms[i,j] != 0 and img_nms[i,j] != 255:
                img_nms[i,j] = 0
    return img_nms


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_b = img[:, :, 1]
    # print(img_b)
    print(img_b.shape, 'img_b.shape')
    data = np.arange(20).reshape(4,5)
    data = np.pad(data,[[1,1],[1,1]],'constant')
    # print(data,'data')
    DT1 = double_threshold_check(data, min_thre=5, max_thre=10)
    print(DT1, 'DT1')
    DT2 = double_threshold_check2(data, min_thre=5, max_thre=10)
    print(DT2, 'DT2')
