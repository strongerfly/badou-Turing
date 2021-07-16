#-*- coding: utf-8 -*-
import numpy as np
import math
import cv2
'''
卷积的原理：卷积的操作简单理解就是通过不同卷积核权重提取特征，通过代码方式描述卷积操作原理
'''
#卷积核3*3*3*3卷积(输出通道，输入通道，filter_h，filter_w),这里是随便定义了3个卷积核，方便使用cv显示卷积后的图像
# conv_filters=np.array([[[[-1,0,1],[-1,0,1],[-1,0,1]],
#                        [[1,0,1],[0,0,0],[-1,0,1]],
#                        [[-0.25,0.25,-0.25],[0.25,1.0,0.25],[0.25,-0.25,0.25]]]])
conv_filters=np.array([
                       [[[-1,0,1],[-2,0,2],[-1,0,1]],
                       [[-1,-2,-1],[0,0,0],[1,2,1]],
                       [[0.25,0.25,0.25],[0.25,0.25,0.25],[0.25,0.25,0.25]]],
                       [[[-1,0,1],[-1,0,1],[-1,0,1]],
                       [[1,0,1],[0,0,0],[-1,0,1]],
                       [[-0.25,0.25,-0.25],[0.25,1.0,0.25],[0.25,-0.25,0.25]]],
                       [[[-1,1,1],[-1,1,1],[-1,1,1]],
                       [[1,1,1],[1,0,0],[-1,1,1]],
                       [[0.1,0.1,0.1],[0.1,1.0,0.1],[0.1,0.1,0.1]]]
                       ])
print(conv_filters.shape)
def conv(input,filters,stride=1,pad=1):
    N,C,H,W=input.shape #这里的N就是mini_batch
    out_chn,input_chn,filter_h,filter_w=filters.shape
    outw=int((W+2*pad-filter_w)/stride+1)
    outh=int((H+2*pad-filter_h)/stride+1)
    conv_result=np.zeros((N,out_chn,outh,outw))
    #填充
    input_data = np.pad(input, [(0,0),(0,0),(pad, pad), (pad, pad)], 'constant')
    #卷积操作，就是图像数据与对应的卷积核相乘再相加操作，这个过程比较耗时,优化可参考im2col实现快速卷积操作，这里仅仅按原理上实现
    for n in range(N):
        for oc in range(out_chn):  # 输出层通道数
            filter = filters[oc]#第oc个卷积核的参数
            for j in range(outh):
                for i in range(outw):
                    #选出将要进行卷积的block
                    conv_block=input_data[n,:,j*stride:j*stride+filter_h,i*stride:i*stride+filter_w]
                    #展平+np.dot来进行相乘相加操作
                    conv_block=conv_block.flatten()
                    filter=filter.flatten().T
                    conv_result[n,oc,j,i]=np.dot(conv_block,filter)
    return conv_result

if __name__=='__main__':
    img = cv2.imread("lenna.png")
    print("img shape ", img.shape)
    h,w,c=img.shape
    newimg=np.zeros(img.shape,dtype=img.dtype)
    # 这是针对stride=1时创建的图像大小，如果stride=2则特征图输出进行创建大小，这里方便演示，用stride=1，卷积核个数也是3方便显示
    showimg=np.zeros((h,2*w,c),dtype=np.uint8)
    showimg[:, 0:w] = img
    #将输入的shape变为（N，C，H，W）形式,
    img=img.transpose(2,0,1)
    img=np.expand_dims(img,axis=0)
    result = conv(img, conv_filters,stride=1)
    result=result.astype(np.uint8)
    print("after conv ,img shape:",result.shape)#(1, 3, 512, 512)
    showimg[:, w:2 * w] = result[0].transpose(1,2,0)
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.imshow("result", showimg)
    cv2.waitKey()




