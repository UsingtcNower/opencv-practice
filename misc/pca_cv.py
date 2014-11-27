import cv2 as cv

def pca(X, numOfPca=0):
    [n,m] = X.shape
    print X
    Pusai = X.mean(axis=0)
    print Pusai
    X = X - Pusai
    print X
    Cov = cv.mulTransposed(X, True)
    
