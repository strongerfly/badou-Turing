import numpy as np


class PCA():
    def __init__(self,n_compontents):
        self.n_compontents = n_compontents

    def fit_transform(self,X):
        X = X - X.mean(axis=0)
        self.covaiance = np.dot(X.T,X)/X.shape[0]
        eig_vals,eig_vectors = np.linalg.eig(self.covaiance)
        idx = np.argsort(-eig_vals)
        self.compontents_ = eig_vectors[:,idx[:self.n_compontents]]
        return np.dot(X,self.compontents_)


pca = PCA(n_compontents=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
newX = pca.fit_transform(X)

print(newX)
