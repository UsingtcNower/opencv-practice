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
        elif num_loops == 1:
            dists = self.compute_distance_one_loops(X)
        elif num_loops == 2:
            dists = self.compute_distance_two_loops(X)
        elif num_loops == 3:
            dists = self.compute_distance_l1(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distance_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i,j] = np.sqrt(np.sum(np.square(X[i,:]-self.X_train[j,:])))
        #print 'dists %r' % dists
        return dists

    def compute_distance_one_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in xrange(num_test):
            dists[i,:] = np.sqrt(np.sum(np.square((X[i,:]-self.X_train)), axis=1))
        #print 'dists %r' % dists
        return dists

    def compute_distance_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        dists = np.multiply(np.dot(X, self.X_train.T),-2)
        #print dists
        s1 = np.sum(np.square(X), axis=1, keepdims=True) #按行求和
        #print s1
        s2 = np.sum(np.square(self.X_train), axis=1) 
        #print s2
        dists = np.add(dists, s1)
        #print dists
        dists = np.add(dists, s2) 
        dists = np.sqrt(dists) 
        #print 'dists %r' % dists
        return dists
        
    def compute_distance_l1(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in xrange(num_test):
            dists[i] = np.sum(np.abs(X[i,:]-self.X_train), axis=1)
        return dists

    def predict_labels(self, dists, k=1):
        
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            closest_y = self.Y_train[np.argsort(dists[i,:])[:k]]
            #print closest_y, np.bincount(closest_y)
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

def main():
    knn = KNearestNeighbor()
    train_X = np.array([[1,2,3],[4,5,6]])
    #print 'train_X %r' % train_X.T
    train_Y = np.array([0,1])
    knn.train(train_X,train_Y)
    X = np.array([[0,1,0],[4,5,6]])
    #print 'X %r' % X
    dists = knn.predict(X,1)
    print 'knn %r' % dists
    dists = knn.predict(X,1,1)
    print '1loop knn %r' % dists
    dists = knn.predict(X,1, 2)
    print '2loop knn %r' % dists

if __name__ == "__main__":
    main()    
