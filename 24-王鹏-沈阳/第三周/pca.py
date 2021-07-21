import numpy as np
class PCA():
    def __init__(self,components):
        self.components = components
    def fit_transform(self,data):
        # covariance
        mean = np.mean(data,axis=0) # column mean
        data = data -mean # center
        covariance = np.dot(data.T, data)/data.shape[0]
        values,vector = np.linalg.eig(covariance)
        index = np.argsort(-values) # descend
        result = vector[:,index[:self.components]] # feature vector after dimensionality reduction
        return np.dot(data,result)
if __name__ =="__main__":
    # 4 features
    data = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])

    pca = PCA(2)
    feature = pca.fit_transform(data)
    print (feature)


