#coding=utf-8
import numpy as np
import cv2
import numpy as np
'''
该文件实现均值hash算法
'''
def GasussNoise(img,mean=0.5,var=0.5,ratio=0.8):
    '''
    img:输入的图像
    mean：高斯均值
    var：高斯方差
    ratio：加入高斯噪声像素点占图像总像素点的比例
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


def aHash(img,scale=8):
    '''
    均值哈希算法
    '''
    #1 图像缩放
    img = cv2.resize(img, (scale, scale), interpolation=cv2.INTER_LINEAR)
    #2.灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #3.对图像求均值
    mean_val=np.mean(gray)
    #4.对大于均值的像素赋值为1，否则为0
    hash_str=np.where(gray>mean_val,1,0).reshape(-1)
    hash_str=''.join(str(i) for i in hash_str)#将array转为str
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
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    print("hash1:{},hash2:{}".format(hash1,hash2))
    n = cmpHash(hash1, hash2)
    print('均值哈希算法相似度：', n)


