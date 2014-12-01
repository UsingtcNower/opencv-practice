import numpy as np

def distance(A,B):
    A = np.asarray(A).flatten()
    B = np.asarray(B).flatten()
    return np.sqrt(np.sum(np.power(A-B,2)))
