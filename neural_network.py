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
			# self.b		= np.ones((self.outputs, 1))
			self.b		= np.ones((1, self.outputs))
		self.d 			= np.zeros(self.inputs)
		
		# # gradient error vector
		# self.g = self.initialize_vector(self.W.shape)

		# # gradient approximation vector
		# self.ga = self.initialize_vector(self.g.shape)

	# Initial values
	def initialize_weights(self):
		if self.init == 'uniform':
			weights = np.random.rand(self.outputs, self.inputs)
		else: #normal
			# weights = np.random.randn(self.outputs, self.inputs)
			weights = np.random.randn(self.inputs, self.outputs)
		return weights

	# Activation functions
	def set_activation(self):
		if self.activation == 'sigmoid':
			self.a = 1 / (1 + np.exp(-self.z))
		elif self.activation == 'relu':
			self.a = max(0, self.z)
		else: #tanh
			self.a = (np.exp(self.z) - np.exp(-self.z)) / (np.exp(self.z) + np.exp(-self.z))
		# print(self.a.shape)
		# print ("a: %s" % (self.a))

	# Output function
	def softmax(self):
		# shift_a to overcome float variable upper bound
		shift_z = self.z - np.max(self.z)
		exp_scores = np.exp(shift_z)
		softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return softmax

	# Derivative of the output function with cross entropy loss
	def softmax_grad(self):
		# s = self.softmax().reshape(-1,1)
		# return np.diagflat(s) - np.dot(s, s.T)
		s = self.softmax()
		s[range(self.y.shape[0]), self.y] -= 1
		# s = s / self.y.shape[0] ???? necessaire?
		return s

	# Derivatives of the activation functions
	def derivative_of_activation(self): 
		if self.activation == 'sigmoid':
			return np.multiply(self.a, (1.0 - self.a))
		elif self.activation == 'relu':
			return 1. * (self.a > 0)
		else: #tanh
			return 1 - np.square(self.a)

	# def update_weights(self, r):
	# 	self.W += -(r*self.g)

	# def check_gradient_computation(self, atol):
	# 	return np.allclose(self.g, self.ga, atol=atol)
		
	def print_layer(self):
		print('W shape ', self.W.shape)
		print ("W:\n %s \n" % (self.W))
		print ("z: %s" % (self.z))
		print ("a: %s" % (self.a))
		#print ("d: %s" % (self.d))
		#print ("g: %s" % (self.g))

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

	return mlp[-1].softmax()

def back_propagation(mlp, loss, learningR, y):

	# Delta over softmax
	mlp[-1].y = y
	delta = loss * mlp[-1].softmax_grad()

	for i in range(len(mlp) - 1, -1, -1):
		# print('layer i =',i)
		if (i == len(mlp) - 1):
			mlp[i].d = delta
		else:
			delta = mlp[i+1].d.dot(mlp[i+1].W.T)
			mlp[i].d = delta * mlp[i].derivative_of_activation()
			# error_sup = np.dot(mlp[i+1].d, mlp[i].W)

		# print('W et delta', mlp[i].W.shape, delta.shape)
		# print('a transpose', mlp[i].a.T.shape)
		# error = np.dot(delta, mlp[i].W.T)
		# error = np.dot(mlp[i].W.T, delta)
		# print('error', error.shape)
		# test = mlp[i].derivative_of_activation()
		# print('delta activation',test.shape)

		# mlp[i].d = np.dot(error, mlp[i].derivative_of_activation())
		# if mlp[i].bias:
		# 	mlp[i].d = mlp[i].d[1:]
		# print('new delta', mlp[i].d.shape)

		# update weights avec learningR

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
			init=array_init, activation=array_activation, layer_id=i+1)
		else:
			# other hidden layers
			l = Layer(array_layers_dim[i], array_layers_dim[i+1], bias=True, \
			init=array_init, activation=array_activation, layer_id=i+1)
	
		net.append(l)

	return net
