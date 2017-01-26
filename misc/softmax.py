#!/usr/bin/python
#coding: utf-8

import numpy as np

def softmax_loss_native(W,X,y,reg):

    dW = np.zeros_like(W)
	dW_each = np.zeros_like(W)
	
    num_train = X.shape[0]
	num_label = W.shape[1]
	f = X.dot(W)
	
	f_max = np.reshape(np.max(f, axis=1),(num_train,1))
	# or f_max = np.max(f, axis=1, keepdims=True)
	prob = np.exp(f-f_max)/np.sum(np.exp(f-f_max), axis=1, keepdims=True)
	y_trueClass = np.zeros_like(prob)
	y_trueClass[np.arrange(num_train), y] = 1.0 #?
	
	loss = 0
	for i in xrange(num_train):
	    for j in xrange(num_label):
		    loss += y_trueClass[i,j]*np.log(prob[i,j])
			dW_each[:,j] = -(y_trueClass[i,j]-prob[i,j])*X[i,:]
			dW += dW_each
	loss /= num_train
	loss += .5*reg*np.sum(W*W)
	dW /= num_train
	
	return loss,dW
	
def softmax_loss_vectorized(W,X,y,reg):
    loss = .0
	dW = np.zeros_like(W)
	num_train = X.shape[0]
	
	f = X.dot(W)
	f_max = np.reshape(np.max(f, axis=1), (num_train,1 ))
	prob = np.exp(f-f_max)/np.sum(np.exp(f-f_max), axis=1, keepdims = True)
	y_trueClass = np.zeros_like(prob)
	y_trueClass[arrange(num_train),y] = 1.0
	loss = np.sum(y_trueClass*np.log(prob)/num_train+0.5*reg*np.sum(W*W))
	dW = -np.dot(X.T,y_trueClass-prob)/num_train+reg*W
	
	return loss,dW
	
	