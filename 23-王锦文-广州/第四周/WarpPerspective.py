#coding=utf-8
import  numpy as np
import cv2
'''
该文件实现图像透视变换
'''
def WarpPerspectiveMatrix(src, dst):
    '''
    :param src:原图上的点
    :param dst:目标图像上对应的点
    :return:透视变换矩阵（至少4个点求解，仿射变换只需要3个点）
    '''
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    #根据公式进行求解方程
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1,0, 0, 0,-A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0,A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    warpMatrix=np.linalg.inv(A).dot(B)
    warpMatrix = warpMatrix.transpose()[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def warp_Interpolation(img,warpMatrix,size=(512,512)):
    '''
    :param img: 要变换的图像
    :param warpMatrix: 变换矩阵
    :return: 变换后的图像
    '''
    srch, srcw,srcchn=img.shape
    dstimg=np.zeros((size[1],size[0],srcchn),dtype=np.uint8)
    dh,dw,dc=dstimg.shape
    #mat=warpMatrix.flatten()
    #mat=np.linalg.inv(warpMatrix)

    #scale_x=dstimg.shape[1]/srcw
    #scale_y=dstimg.shape[0]/srch

    # for j in range(dh-1):
    #     for i in range(dw-1):
    #         num=(mat[7]-mat[8]*j)*(mat[5]*i-mat[3])+(mat[5]*j-mat[4])*(mat[8]*i-mat[6])
    #         den=(mat[2]*j-mat[1])*(mat[5]*i-mat[3])+(mat[2]*i-mat[0])*(mat[7]*j-mat[4])
    #         srcx=num/(den+1e-8)
    #         num1=(mat[6]-mat[8]*i)-(mat[2]*i-mat[0])*srcx
    #         den1=mat[5]*i-mat[3]
    #         srcy=num1/(den1+1e-8)
    #
    #         if srcx>0 and srcx<srcw and srcy>0 and srcy<srch:
    #             dstimg[j,i]=img[int(srcy),int(srcx)]
    #这里简单根据公式进行计算了，插值也直接用原始图像像素值插值到对应位置的新图像上了,
    for j in range(srch-1):
        for i in range(srcw-1):
            tmp=np.array([i,j,1])
            dstxyz=warpMatrix.dot(tmp)
            dstx,dsty,dstz=dstxyz
            dstx/=dstz
            dsty/=dstz
            if dstx > 0 and dstx < dw and dsty > 0 and dsty < dh:
                dstimg[int(dsty), int(dstx)] = img[j, i]
    dstimg.astype(np.uint8)

    return  dstimg

if __name__ == '__main__':
    img = cv2.imread('lenna.png',1)
    h, w, c = img.shape
    p1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [h - 1, h - 1]])
    p2 = np.float32([[w*0.05, h * 0.1], [w * 0.9, h * 0.2], [w * 0.1, h * 0.85], [w * 0.85, h * 0.9]])
    #方法一 根据公式计算透视变换矩阵，并计算透视变换后图像
    warpMatrix = WarpPerspectiveMatrix(p1, p2)#此函数等价于opencv的cv2.getPerspectiveTransform求出的结果一致
    #print("Ma:",warpMatrix)
    dstimg=warp_Interpolation(img, warpMatrix,size=(w, h))#对图像进行透视变换处理，并插值，这里直接用最近邻插值了


    ####方法二 以下是使用opencv实现的
    M = cv2.getPerspectiveTransform(p1, p2)
    dst = cv2.warpPerspective(img, warpMatrix, (w, h))
    cv2.imshow('src_img', img)
    cv2.imshow("my_result_warp_img", dstimg)
    cv2.imshow('cv_result_warp_img', dst)
    print("my warpmatrix:{},opencv warpmatrix:{}".format(warpMatrix,M))
    cv2.waitKey(0)



