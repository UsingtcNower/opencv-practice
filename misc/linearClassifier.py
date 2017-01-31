from svm import *
from softmax import *

class LinearClassifer(object):
    def __init__(self):
	    self.W = None
	
	'''
	Input:
	@param X (N,D) train data
	@param y (N,) labels
	
	Output:
	loss lists
	'''
	def train(self, X, y, lr=1e-3, reg=1e-5, iters=100,
	                batch_size=200, verbose=True):
		num_train,dim = X.shape
		num_label = np.max(y)+1 #assume y takes 0,1...C-1 where C is the number of class
		
		if self.W == None:
		    self.W = 0.001*np.random.randn(dim, num_label)
		
		loss_history = []
		for i in xrange(iters):
			sample_index = np.random.choice(num_train, batch_size, replace=False)
			X_batch = X[sample_index,:]
			y_batch = y[sample_index]
			#get loss, gradient
			loss, grad = self.loss(X_batch, y_batch, reg)
			loss_history.append(loss)
			
			self.W -= lr*grad
			if verbose and i%100 == 0:
			    print 'Iteration %d/%d: loss %f' % (i,iters, loss)
		
		return loss_history
	
	'''
	@param X (N,D) 
	
	Output:
	predict label vector
	'''
	def predict(self,X):
	    y_pred = np.argmax(X.dot(self.W),axis=1)
		return y_pred
		
	def loss(self, X, y, reg):
	    pass
	
class LinearSVM(LinearClassifer):
    def loss(self,X,y,reg):
	    return svm_loss_vectorized(self.W, X, y, reg)
		
class Softmax(LinearClassifer):
    def loss(self.W, X, y, reg):
	    return softmax_loss_vectorized(self.W, X, y, reg)