#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

import numpy as np

class Layer:

	def __init__(self, inputs, outputs, bias, init, activation, optimizer, dropout, layer_id):
	   
		self.inputs     = inputs
		self.outputs    = outputs
		self.init 		= init
		self.activation = activation
		self.layer_id   = layer_id
		self.bias		= bias
		self.optimizer	= optimizer
		self.dropout	= dropout
		# init vectors / z = x * W(eights) + b(ias) with x == inputs
		# layer result a = activation(z)
		self.W 			= self.initialize_weights()
		self.z 			= np.zeros(self.inputs)
		self.a 			= np.zeros(self.inputs)
		self.b			= np.zeros(self.outputs)
		if self.bias:
			self.b		= np.ones((1, self.outputs))
		self.dW			= np.zeros(self.inputs)
		self.db			= np.zeros(self.outputs)
		if optimizer == 'adam':
			# each layer has individual weights/bias modifiers and inertial 'learning rate'
			self.M 		= np.zeros((self.inputs, self.outputs))
			self.R 		= np.zeros((self.inputs, self.outputs))
			if self.bias:
				self.Mb	= np.zeros((1, self.outputs))
				self.Rb	= np.zeros((1, self.outputs))
		if self.dropout == True:
			self.mask 		= np.zeros(self.inputs)

	# Initial values
	def initialize_weights(self):
		if self.init == 'uniform':
			weights = np.random.rand(self.inputs, self.outputs)
		else: #normal
			weights = np.random.randn(self.inputs, self.outputs)
		return weights

	# Activation functions
	# softmax is not really an 'activation' but can fit here
	def set_activation(self):
		if self.activation == 'sigmoid':
			self.a = 1 / (1 + np.exp(-self.z))
		elif self.activation == 'tanh':
			self.a = (np.exp(self.z) - np.exp(-self.z)) / (np.exp(self.z) + np.exp(-self.z))
		elif self.activation == 'softmax':
			# shift_a to overcome float variable upper bound
			shift_z = self.z - np.max(self.z, axis=1, keepdims=True)
			exp_scores = np.exp(shift_z)
			self.a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		else: #relu
			self.a = self.z * (self.z > 0)

	# Derivatives of the activation functions
	def derivative_of_activation(self): 
		if self.activation == 'sigmoid':
			return np.multiply(self.a, (1.0 - self.a))
		elif self.activation == 'tanh':
			return 1 - np.square(self.a)
		elif self.activation == 'softmax':
			s = self.a
			s[range(self.y.shape[0]), self.y] -= 1
			s = s / self.y.shape[0]
			return s
		else: #relu
			return 1. * (self.a > 0)

# log loss: loss is very high if yhat far from y (on 0/1 scale)
def cross_entropy_loss(yhat, y):
	log_likelihood = -np.log(yhat[range(y.shape[0]), y])
	loss = np.sum(log_likelihood) / y.shape[0]
	return loss

def feed_forward(mlp, x, y, predict):

	# z = x * W(eights) + b(ias) with x == inputs
	# layer result a = activation(z)

	mlp[-1].y = y
	# p useful only for dropout: it's the dropout rate
	p = 0.8

	for i in range(len(mlp)):

		if (i == 0):
			#first layer: data input x
			data = x
		else:
			data = mlp[i-1].a
		
		if predict == False and mlp[i].dropout == True:
			mask = (np.random.rand(*data.shape) < p) / p
			data = data * mask
			mlp[i].mask = mask

		mlp[i].z = np.dot(data, mlp[i].W) + mlp[i].b

		mlp[i].set_activation()

	return mlp[-1].a

def back_propagation(mlp, learningR, regL2, x, iteration):

	delta = 1
	# for adam
	epsilon = 0.00000001
	beta1 = 0.9
	beta2 = 0.999

	for i in range(len(mlp) - 1, -1, -1):

		dz = mlp[i].derivative_of_activation() * delta

		if mlp[i].dropout == True:
			dz = dz * mlp[i].mask

		if (i == 0):
			#first layer: data input x
			dW = np.dot(x.T, dz)
		else:
			dW = np.dot(mlp[i-1].a.T, dz)
			
		delta = np.dot(dz, mlp[i].W.T)
		db = np.sum(dz, axis=0)

		mlp[i].dW = dW
		mlp[i].db = db

	# update weights avec  learning rate
	for i in range(len(mlp)):
		if regL2 != 0.:
			mlp[i].dW += regL2 * mlp[i].W 
		if mlp[i].optimizer == 'adam':
			# for W
			mlp[i].M = beta1 * mlp[i].M + (1. - beta1) * mlp[i].dW
			mlp[i].R = beta2 * mlp[i].R + (1. - beta2) * np.square(mlp[i].dW)
			m_k_hat = mlp[i].M / (1. - beta1**(iteration+1))
			r_k_hat = mlp[i].R / (1. - beta2**(iteration+1))
			mlp[i].W += -learningR / (np.sqrt(r_k_hat) + epsilon) * m_k_hat
			# for b
			mlp[i].Mb = beta1 * mlp[i].Mb + (1. - beta1) * mlp[i].db
			mlp[i].Rb = beta2 * mlp[i].Rb + (1. - beta2) * np.square(mlp[i].db)
			m_k_hat = mlp[i].Mb / (1. - beta1**(iteration+1))
			r_k_hat = mlp[i].Rb / (1. - beta2**(iteration+1))
			mlp[i].b += -learningR / (np.sqrt(r_k_hat) + epsilon) * m_k_hat
		else:
			mlp[i].W += -learningR * mlp[i].dW
			mlp[i].b += -learningR * mlp[i].db

def topology(mlp):
	layers = []
	for i in range(len(mlp)):
		layer = [mlp[i].W.shape[0], mlp[i].W.shape[1], mlp[i].bias, \
		mlp[i].init, mlp[i].activation, mlp[i].optimizer]
		layers.append(layer)
	return layers

def net_loader(layers):
	net = []
	for i in range(len(layers)):
		l = Layer(layers[i][0], layers[i][1], layers[i][2], \
			layers[i][3], layers[i][4], layers[i][5], False, i)
		net.append(l)
	return net

def net_constructer(features, categories, array_layers_dim, array_init, array_activation, optimizer, dropout):
	net = []

	for i in range(len(array_layers_dim)):
		# first layer connected to all features
		if (i == 0):
			l = Layer(features, array_layers_dim[i], bias=True,  \
				init=array_init, activation=array_activation, optimizer=optimizer, dropout=False, layer_id=i)
			net.append(l)

		if i == len(array_layers_dim) - 1:
			# output layer
			l = Layer(array_layers_dim[i], categories, bias=True, \
			init=array_init, activation='softmax', optimizer=optimizer, dropout=False, layer_id=i+1)
		else:
			# other hidden layers
			l = Layer(array_layers_dim[i], array_layers_dim[i+1], bias=True, \
			init=array_init, activation=array_activation, optimizer=optimizer, dropout=dropout, layer_id=i+1)
	
		net.append(l)

	return net
