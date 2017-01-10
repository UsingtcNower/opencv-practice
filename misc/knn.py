#!/usr/bin/python
#!coding:utf-8
import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distance_no_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distance_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        dists = np.multiply(np.dot(X, self.X_train.T),-2)
        s1 = np.sum(np.square(X), axis=1, keepdims=True) #按行求和
        #s2 = np.sum(np.square(self.X_train.T), axis=1) 
        #dists = np.add(s1)
        #dists = np.add(s2) 
        #dists = np.square(dists) 
        return s1

def main():
    knn = KNearestNeighbor()
    train_X = np.array([[1,2,3],[4,5,6]])
    #print 'train_X %r' % train_X.T
    train_Y = np.array([0])
    knn.train(train_X,train_Y)
    X = np.array([[0,1,0],[0,0,1]])
    #print 'X %r' % X
    dists = knn.compute_distance_no_loops(X)
    print 'dists %r' % dists

if __name__ == "__main__":
    main()    
