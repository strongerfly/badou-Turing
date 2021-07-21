import numpy as np

def WarpPerspectiveMatrix(src,dst):
    # 原矩阵和目标矩阵的行数相等，并且行数大于4才是正常情况，否则就终止
    assert  src.shape[0]==dst.shape[0] and src.shape[0]>=4
    nums = src.shape[0]
    A=np.zeros((nums*2,8))
    B = np.zeros((nums * 2, 1))
    for i in range(nums):
        A_i=src[i]
        B_i=dst[i]
        A[2*i,:]=[A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        A[2*i+1,:]=[0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i,:]=[B_i[0]]
        B[2 * i+1, :] = [B_i[1]]

    A=np.mat(A)
    # A*martrix=B   ----> A*A.I*martrix=A.I*B=martrix
    martrix=A.I*B

    martrix=np.insert(martrix,martrix.shape[0],1,axis=0)
    martrix=martrix.reshape((3,3))
    return martrix

if __name__=='__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)




