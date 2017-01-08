#!/usr/bin/python

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

        return self.prefict?_labels(dists, k=k)

    def compute_distance_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        
