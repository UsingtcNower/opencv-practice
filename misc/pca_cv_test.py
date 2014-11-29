import pca_cv
import numpy as np

I1 = np.asmatrix([1,1])
I2 = np.asmatrix([1,0])
I3 = np.asmatrix([0,1])
I4 = np.asmatrix([3,3])

A = I1.T
A = np.hstack((A,I2.T))
A = np.hstack((A,I3.T))
A = np.hstack((A,I4.T))
print A
print pca_cv.pca(A,2)
