#coding=utf-8
import numpy as np
import cv2
import numpy as np
'''
该文件实现差值hash算法
'''
def GasussNoise(img,mean=0.1,var=0.5,ratio=0.9):
    '''
    img:输入的图像
    mean：高斯均值
    var：高斯方差
    ratio：加入高斯噪声像素点占图像总像素点的比例
    该函数实现高斯噪声，主要目的是用于对比，加入噪声前后的hash变化，由于高斯噪声比较简单，这里直接将该函数放这里了，不另外做一个py文件了
    '''
    h,w,c=img.shape
    noise_num=int(h*w*ratio)
    #随机取图像的索引点
    X=np.random.choice(range(w),size=noise_num)
    Y=np.random.choice(range(h),size=noise_num)
    #归一化0-1之间在操作
    img=img/255.0
    #高斯分布
    noise = np.random.normal(mean, var, noise_num)
    dst_img=img.copy()
    for x,y,n in zip(list(X),list(Y),list(noise)):
        dst_img[y,x,:]=img[y,x,:]+n
    dst_img = np.clip(dst_img, 0, 1)
    dst_img = np.uint8(dst_img * 255)
    return  dst_img


def dHash(img,scale=8):
    '''
    差值哈希算法
    '''
    scale_w, scale_h=scale+1,scale
    # 1.图像缩放
    img = cv2.resize(img, (scale_w,scale_h), interpolation=cv2.INTER_LINEAR)
    #2. 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    gray=gray.reshape(-1)#变为1维数组
    for i,(j1,j2) in enumerate(zip(gray,gray[1:])):
        if (i+1)%scale_w==0:#换行的值不做比较（上一行末尾和下一行开头的两个值不做比较）
            continue
        if j1>j2:#如果前一个元素值大于后一个元素，则赋值为1
            hash_str = hash_str + '1'
        else:
            hash_str= hash_str + '0'

    return hash_str

def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for j1,j2 in zip(hash1,hash2):#比较相似度，统计不相等的值次数
       if j1!=j2:
           n+=1
    return n

if __name__=='__main__':
    img1 = cv2.imread('lenna.png')
    img2=GasussNoise(img1)#加入高斯噪声
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    print("hash1:{},hash2:{}".format(hash1,hash2))
    n = cmpHash(hash1, hash2)
    print('差值哈希算法相似度：', n)