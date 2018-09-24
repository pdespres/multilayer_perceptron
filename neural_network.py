#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

import numpy as np

class Layer:

	def __init__(self, inputs, outputs, bias, init, activation, layer_id):
	   
		self.inputs     = inputs
		self.outputs    = outputs
		self.init 		= init
		self.activation = activation
		self.layer_id   = layer_id
		self.bias		= bias
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

	# Initial values
	def initialize_weights(self):
		if self.init == 'uniform':
			weights = np.random.rand(self.outputs, self.inputs)
		else: #normal
			weights = np.random.randn(self.inputs, self.outputs)
		return weights

	# Activation functions
	# softmax is not really an 'activation' but can fit here
	def set_activation(self):
		if self.activation == 'sigmoid':
			self.a = 1 / (1 + np.exp(-self.z))
		elif self.activation == 'relu':
			self.a = max(0, self.z)
		elif self.activation == 'softmax':
			# shift_a to overcome float variable upper bound
			shift_z = self.z - np.max(self.z)
			exp_scores = np.exp(shift_z)
			self.a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		else: #tanh
			self.a = (np.exp(self.z) - np.exp(-self.z)) / (np.exp(self.z) + np.exp(-self.z))

	# Derivatives of the activation functions
	def derivative_of_activation(self): 
		if self.activation == 'sigmoid':
			return np.multiply(self.a, (1.0 - self.a))
		elif self.activation == 'relu':
			return 1. * (self.a > 0)
		elif self.activation == 'softmax':
			s = self.a
			s[range(self.y.shape[0]), self.y] -= 1
			s = s / self.y.shape[0]
			return s
		else: #tanh
			return 1 - np.square(self.a)

	def print_layer(self):
		print('W shape ', self.W.shape)
		print ("W:\n %s \n" % (self.W))
		print ("z: %s" % (self.z))
		print ("a: %s" % (self.a))

def cross_entropy_loss(yhat, y):
	log_likelihood = -np.log(yhat[range(y.shape[0]), y])
	loss = np.sum(log_likelihood) / y.shape[0]
	return loss

def feed_forward(mlp, x):

	# z = x * W(eights) + b(ias) with x == inputs
	# layer result a = activation(z)
	for i in range(len(mlp)):

		if (i == 0):
			#first layer: data input x
			data = x
		else:
			data = mlp[i-1].a

		mlp[i].z = np.dot(data, mlp[i].W) + mlp[i].b
		
		mlp[i].set_activation()

	return mlp[-1].a

def back_propagation(mlp, learningR, x, y):

	mlp[-1].y = y
	delta = 1

	for i in range(len(mlp) - 1, -1, -1):

		dz = mlp[i].derivative_of_activation() * delta

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
		mlp[i].W += -learningR * mlp[i].dW
		mlp[i].b += -learningR * mlp[i].db

def net_constructer(features, categories, array_layers_dim, array_init, array_activation):
	net = []

	for i in np.arange(len(array_layers_dim), dtype=int):

		# first layer connected to all features
		if (i == 0):
			l = Layer(features, array_layers_dim[i], bias=True,  \
				init=array_init, activation=array_activation, layer_id=i)
			net.append(l)

		if i == len(array_layers_dim) - 1:
			# output layer
			l = Layer(array_layers_dim[i], categories, bias=True, \
			init=array_init, activation='softmax', layer_id=i+1)
		else:
			# other hidden layers
			l = Layer(array_layers_dim[i], array_layers_dim[i+1], bias=True, \
			init=array_init, activation=array_activation, layer_id=i+1)
	
		net.append(l)

	return net
