import cv2
import numpy as np
import time

#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str
 
#差值感知算法
def dHash(img):
    #缩放8*9
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str
 
#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n

def hashDemo():
    img1=cv2.imread('../images/lenna.png')
    img2=cv2.imread('../images/lenna_noise.png')
    ahash1= aHash(img1)
    ahash2= aHash(img2)
    print(ahash1)
    print(ahash2)
    n=cmpHash(ahash1,ahash2)
    print('均值哈希算法相似度：',n)

    dhash1= dHash(img1)
    dhash2= dHash(img2)
    print(dhash1)
    print(dhash2)
    n=cmpHash(dhash1,dhash2)
    print('差值哈希算法相似度：',n)
    return ahash1,ahash2,dhash1,dhash2

def myHashDemo():
    # 均值哈希算法
    def aHash(img):
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)# 缩放为8*8
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转换为灰度图
        avg_gray = np.sum(gray)/64 #平均灰度
        mask = (gray > avg_gray).astype(np.int)
        hash_str = [str(x) for x in mask.reshape((-1,)).tolist()]
        hash_str = ''.join(hash_str)
        return hash_str

    # 差值哈希算法
    def dHash(img):
        # 缩放8*9
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
        # 转换灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pre = gray[:,:8]
        next = gray[:,1:]
        mask = (pre > next).astype(np.int)
        hash_str = [str(x) for x in mask.reshape((-1,)).tolist()]
        hash_str = ''.join(hash_str)
        return hash_str

    img1 = cv2.imread('../images/lenna.png')
    img2 = cv2.imread('../images/lenna_noise.png')
    ahash1 = aHash(img1)
    ahash2 = aHash(img2)
    print(ahash1)
    print(ahash2)
    n = cmpHash(ahash1, ahash2)
    print('均值哈希算法相似度：', n)

    dhash1 = dHash(img1)
    dhash2 = dHash(img2)
    print(dhash1)
    print(dhash2)
    n = cmpHash(dhash1, dhash2)
    print('差值哈希算法相似度：', n)
    return ahash1,ahash2,dhash1,dhash2

if __name__=='__main__':
    t0 = time.time()
    ahash1,ahash2,dhash1,dhash2 = hashDemo()
    t1 = time.time()
    print('-'*60)
    my_ahash1,my_ahash2,my_dhash1,my_dhash2 = myHashDemo()
    t2 = time.time()
    print('校验:',ahash1 == my_ahash1,ahash2 == my_ahash2,dhash1 == my_dhash1,dhash2 == my_dhash2)
    print('time cmp:',t1 - t0,t2 - t1,',ratio:',(t2 - t1) / (t1 - t0))
