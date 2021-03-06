#!/usr/bin/python
#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
	    self.params = {}
		self.params['w1'] = np.random.randn(input_size, hidden_size)*std
		self.params['b1'] = np.zeros(hidden_size)
		self.params['w2'] = np.random.randn(hidden_size, output_size)*std
		self.params['b2'] = np.zeros(output_size)
		
	'''
	@param X (N,D)
	'''
	def predict(self,X):
	  y_pred = None
		h1 = np.maximum(0, X.dot(self.params['w1'])+self.params['b1']) #relu
		scores = h1.dot(self.params['w2'])+self.params['b2']
		y_pred = np.argmax(scores, axis=1)
		return y_pred
	
	'''
	@param X (N,D)
	@param y (N,)
	@param reg
	
	outputs:
	loss, grads
	'''
	def loss(self, X, y, reg):
		  w1 = self.params['w1']
		  b1 = self.params['b1']
		  w2 = self.params['w2']
		  b2 = self.params['b2']
	    #forward
	    h1 = np.maximum(0, X.dot(w1)+b1) #relu
	    score = h1.dot(w2)+b2
	    ##softmax
	    prob = np.exp(score)/np.sum(np.exp(score))
	    loss = -np.log(prob)
	    #backward
	    #暂时使用网上的微分公式
	    dscores = probs  # (N,C) 
      dscores[range(N), y] -= 1  #  这个是输出的误差敏感项也就是梯度的计算，具体可以看上面softmax 的计算
      dscores /= N
      # Backprop into W2 and b2
      dW2 = np.dot(h1.T, dscores)  # (H,C) BP算法的计算，下面同理
      db2 = np.sum(dscores, axis=0, keepdims=True)  # (1,C
      # Backprop into hidden layer
      dh1 = np.dot(dscores, W2.T)  # (N,H)
      # Backprop into ReLU non-linearity
      dh1[h1 <= 0] = 0
      # Backprop into W1 and b1
      dW1 = np.dot(X.T, dh1)  # (D,H)
      db1 = np.sum(dh1, axis=0, keepdims=True)  # (1,H)
      # Add the regularization gradient contribution
	    
	'''
	@param X (N,D)
	@param y (N,) 0=<yi<C
	@param X_val (N_val, D)
	@param y_val (N_val, )
	@param lr learning_rate
	@param lr_decay decay learning_rate
	@param reg regularize
	@param iter iter times
	@param batch_size batch size
	@param verbose print log
	'''
	def train(self, X, y, X_val, y_val,
	                lr=1e-3, lr_decay=0.95,
					reg=1e-5, iters=100,
					batch_size=200, verbose=False):
		num_train, dim = X.shape
		loss_history = []
		iters_per_epoch = np.max(num_train/batch_size,1)
		
		for i in xrange(iters):
		  sample_index = np.random.choice(num_train, batch_size, replace=False)
			X_batch = X[sample_index,:]
			y_batch = y[sample_index]
			loss, gradient = self.loss(X_batch, y_batch, reg)
			loss_history.append(loss)
			# sgd update 
			self.params['b2'] += -lr*gradient['b2']
			self.params['w2'] += -lr*gradient['w2']
			self.params['b1'] += -lr*gradient['b1']
			self.params['w1'] += -lr*gradient['w1']
			
			if verbose and i%100==0:
			    print 'iteration %d/%d: loss %d' % (i,iters, loss)
			
			if i%iters_per_epoch==0:
			    lr *= lr_decay
			
		return loss_history