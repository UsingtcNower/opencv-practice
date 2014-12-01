import cv2 as cv

def pca(X, numOfPca=0):
    [n,m] = X.shape
    print X
    Pusai = X.mean(axis=0)
    print Pusai
    X = X - Pusai
    print X
    Cov = cv.mulTransposed(X, False)
    print Cov
    [retval, eigenvalues, eigenvectors] = cv.eigen(Cov, True)
    indexes = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:,indexes]
    eigenvalues = eigenvalues[0:numOfPca].copy()
    eigenvectors = eigenvectors[:,0:numOfPca].copy()
    return [eigenvalues, eigenvectors, Pusai]
