#!/usr/bin/python
#coding: utf-8

def svm_loss_native(W,X,y,reg):

    loss = .0
	num_train = X.shape[0]
	num_label = W.shape[1]
	
	dW = np.zeros_like(W)
	score = X.dot(W)
	y_trueClass = np.zeros_like(score)
	y_trueClass[np.arrange(num_train), y] = 1.0
	for i in xrange(num_train):
	    correct_class_score = score[i, y[i]]
	    for j in xrange(num_label):
		    if j == y[i]:
			    continue
			margin =  score[i,j]-correct_class_score+1
			if margin > 0:
			    loss += margin
				dW[:,y[i]] += -X[i,:]
				dW[:,j] += X[i,:]
	
	loss /= num_train
	loss += .5*reg*np.sum(W*W)
	
	dW /= num_train
	dW += .5*reg*W
	
	return loss
		    
			
def svm_loss_vectorized(W,X,y,reg):
    loss = .0
	num_train = X.shape[0]
	num_label = W.shape[1]
	
	score = X.dot(W)
	dW = np.zeros_like(W)
	y_trueClass = np.zeros((num_train, num_label))
	y_trueClass[range(num_train),y] = 1.0
	correct_class_score = score[range(num_train),y]
	correct_class_score = np.reshape(correct_class_score, (num_train,1))
	margin = score - correct_class_score+1
	margin[margin<=0] = 0
	loss += np.sum(margin)
	loss /= num_train
	loss += .5*reg*np.sum(W*W)
	
	margin[margin>0] = 1.0
	row_sum = np.sum(margin, axis=1)
	margin[range(num_train),y] = -row_sum
	dW += np.dot(X.T,margin)/num_train+reg*W