import numpy as np

def project(A,B,Pusai=None):
    if Pusai is None:
        return np.dot(A,B)
    return np.dot(A-Pusai,B)

def pca(X,Y,numOfPca=0):
    [n,m] = X.shape
    Pusai = X.mean(axis=0)
    X = X - Pusai
    Cov = np.dot(X,X.T)
    [eigenvalues, eigenvectors] = np.linalg.eigh(C)
    indexes = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:,indexes]
    eigenvalues = eigenvalues[0:numOfPca].copy()
    eigenvectors = eigenvectors[:,0:numOfPca].copy()
    return [eigenvalues, eigenvectors, Pusai]
