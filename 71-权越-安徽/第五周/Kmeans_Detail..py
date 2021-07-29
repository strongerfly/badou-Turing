import random
import math
import numpy as np

# 手推kmean聚合过程

def function(data):
    w,h=data.shape
    # 获得随机点数组下标
    r1=random.randint(0,w-1)
    r2=random.randint(0, w-1)
    if(r2==r1):
        r2 = random.randint(0, w - 1)
    # 获得初始化中心点
    p1=data[r1]
    p2=data[r2]

    # 最大循环20次结束查找
    for i in range(20):
        lst1 = []
        lst2 = []
        for i in range(0, w):
            x, y = data[i]
            dst1 = math.sqrt(math.pow((x - p1[0]), 2) + math.pow((y - p1[1]), 2))
            dst2 = math.sqrt(math.pow((x - p2[0]), 2) + math.pow((y - p2[1]), 2))
            if (dst1 < dst2) :
                lst1.append(data[i])
            else:
                lst2.append(data[i])

        # 获得新的集合质心
        p1_new = np.mean(lst1, axis=0)
        p2_new= np.mean(lst2, axis=0)

        # 早停条件 ，如果质心和上一次质心相等则停止
        if((p1==p1_new).all() and (p2==p2_new).all()):
            break;
        else:
            p1=p1_new
            p2=p2_new

    return lst1,lst2

if __name__=="__main__":
    data=np.array([[0,0],[1,2],[3,1],[8,8],[9,10],[10,7]])
    lst1,lst2=function(data)
    print("lst1",lst1)
    print("lst2", lst2)