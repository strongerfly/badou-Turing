import numpy as np

class PCA():
    def __init__(self,n_components):
        self.n_components=n_components

    def fit_transform(self,X):
        self.n_features=X.shape[1]
        X=X-X.mean(axis=0)

        self.covariance=np.dot(X.T,X)/X.shape[0]

        eig_vals,eig_vector=np.linalg.eig(self.covariance)

        idx=np.argsort(-eig_vals)

        self.components=eig_vector[:,idx[:self.n_components]]

        return np.dot(X,self.components)
PCA=PCA(n_components=2)

# X=np.array([[-2,-3,-9],[1,3,4],[3,6,9]],[[-2,-3,-9],[1,3,4],[3,6,9]],[[-2,-3,-9],[1,3,4],[3,6,9]])
X=np.array([[-1,2,66,-1],[-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
x=PCA.fit_transform(X)
print(x.shape)
print(x)