import numpy as np

def project(A,B,Pusai=None):
    if Pusai is None:
        return np.dot(A,B)
    return np.dot(A-Pusai,B)

def pca(X,numOfPca=0):
    [n,m] = X.shape
    Pusai = X.mean(axis=0)
    print 'mean'
    print Pusai
    X = X - Pusai
    Cov = np.dot(X,X.T)
    print 'Cova'
    print Cov
    [eigenvalues, eigenvectors] = np.linalg.eigh(Cov)
    indexes = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:,indexes]
    eigenvalues = eigenvalues[0:numOfPca].copy()
    eigenvectors = eigenvectors[:,0:numOfPca].copy()
    return [eigenvalues, eigenvectors, Pusai]
