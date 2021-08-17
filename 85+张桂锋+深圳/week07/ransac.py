'''
第七周作业：
1）ransac实现
2）hash实现
'''
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import random
from sklearn.metrics import mean_squared_error
import numpy as np
trans=PolynomialFeatures(degree=2, include_bias=False)
linear_clf=LinearRegression()
def ransac(data,model,n,k,t,d):
    '''
    data:二维数据
    model:训练好的模型
    n:随机抽取样本数目
    k:迭代次数
    t:阈值
    d:测试集大于多少才认为其是好的模型
    '''
    data_size=len(data)
    epoch=0
    best_model=None
    besterr=np.inf
    best_inlier_idxs=None
    while epoch<k:
        #print(epoch)
        maybe_idxs,test_idxs=shuffle_data(data_size, n)
        #print(maybe_idxs,test_idxs)
        maybe_inliers=data[maybe_idxs,:]
        #print(maybe_inliers)
        test_inlier=data[test_idxs,:]
        #print(test_inlier)
        maybeModel=linear_clf.fit(maybe_inliers[:,:-1],maybe_inliers[:,-1])
        #y_pred_maybe = maybeModel.predict(test_inlier[:, :-1])
        #y_test_maybe = test_inlier[:, -1]
        #print(y_pred_maybe,y_test_maybe)
        test_error,_=get_error(maybeModel, test_inlier[:,:-1], test_inlier[:,-1])
        #print(test_error)
        also_idxs=test_idxs[test_error<t]
        #print(also_idxs)
        also_inliers=data[also_idxs,:]
        #print(len(also_inliers))
        if len(also_inliers)>d:
            better_data=np.concatenate((maybe_inliers,also_inliers))
            betterModel=linear_clf.fit(better_data[:,:-1],better_data[:,-1])
            _,thisError=get_error(betterModel,better_data[:,:-1],better_data[:,-1])
            if thisError<besterr:
                best_model=betterModel
                besterr=thisError
                best_inlier_idxs=np.concatenate((maybe_idxs,also_idxs))
        epoch+=1
    if best_model is None:
        raise ValueError("无法拟合出model")
    else:
    	return best_model,besterr,best_inlier_idxs
def shuffle_data(data_row,n):
    idxs=np.arange(data_row)
    np.random.shuffle(idxs)
    return idxs[:n],idxs[n:]
def get_error(model,test,y_true):
    y_predict=model.predict(test)
    #print(test)
    #print(y_predict)
    #print(y_true)
    error=np.sqrt((y_predict-y_true)**2)
    #print(error)
    mean_error=np.sqrt(mean_squared_error(y_true, y_predict))
    #mean_error=np.mean(error)
    return error,mean_error

if __name__ == '__main__':
    #points = np.array([[1,2,4], [5,7,11], [8,9,18], [9,10,22], [10,11, 19], [19,13,37],[3,7,2],[2,2,1],[8,7,6],[11,53,21]])
    points = 100*random.random(size=(1000,20))#随机生成数组
    print("---------------------------------直接使用----------------------------------------")
    print("---------------------------------X变量----------------------------------------")
    print(points[:,:-1])
    print("---------------------------------Y变量----------------------------------------")
    print(points[:,-1])
    betterModel=linear_clf.fit(points[:,:-1],points[:,-1])
    #betterModel=linear_clf.fit(trans.fit_transform(points[:,:-1]),points[:,-1])
    #print(trans.fit_transform(points[:,:-1]))
    y_pred = betterModel.predict(points[:,:-1])
    y_test =points[:,-1]
    print(y_pred,y_test)
    print("---------------------------------权重----------------------------------------")
    print(betterModel.coef_)
    print("---------------------------------截距----------------------------------------")
    print(betterModel.intercept_)
    print("---------------------------------误差----------------------------------------")
    print('RMSE为：',np.sqrt(mean_squared_error(y_test,y_pred)))
    print("---------------------------------代入计算结果----------------------------------------")
    print(points[:,:-1].dot(betterModel.coef_)+betterModel.intercept_)
    print("---------------------------------ransac实现----------------------------------------")
    best_model,besterr,best_inlier_idxs=ransac(points,linear_clf,600,1000,50,350)
    print(best_model.coef_)
    print(best_model.intercept_)
    print(besterr)
