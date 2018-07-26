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

		# init vectors / z = x * W(eights) + b(ias) with x == inputs / layer result a = activation(z)
		self.W 			= self.initialize_weights()
		self.z 			= np.zeros(self.inputs)
		self.a 			= np.zeros(self.inputs)
		self.bias		= np.zeros(self.inputs)
		if bias:
			self.bias	= np.ones((self.outputs, 1))
		self.d 			= np.zeros(bias + self.inputs)
		
		# # gradient error vector
		# self.g = self.initialize_vector(self.W.shape)

		# # gradient approximation vector
		# self.ga = self.initialize_vector(self.g.shape)

	def initialize_weights(self):
		if self.init == 'uniform':
			weights = np.random.rand(self.outputs, self.inputs)
		else: #normal
			weights = np.random.randn(self.outputs, self.inputs)
		return weights

	def set_activation(self):
		if self.activation == 'sigmoid':
			self.a = 1 / (1 + np.exp(-self.z))
		elif self.activation == 'relu':
			self.a = max(0, self.z)
		else: #tanh
			self.a = (np.exp(self.z) - np.exp(-self.z)) / (np.exp(self.z) + np.exp(-self.z))

	def softmax(self):
		shift_a = self.a - np.max(self.a)
		exp_scores = np.exp(shift_a)
		return exp_scores / np.sum(exp_scores, axis=0)

	def softmax_grad(self):
		s = softmax.reshape(-1,1)
		return np.diagflat(s) - np.dot(s, s.T)
//////////SOFTMAX?
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
		print ("W:\n %s \n" % (self.W))
		print ("z: %s" % (self.z))
		print ("a: %s" % (self.a))
		#print ("d: %s" % (self.d))
		#print ("g: %s" % (self.g))

def cross_entropy_loss(yhat, y):
	log_likelihood = -np.log(yhat.T[range(y.shape[0]), y])
	loss = np.sum(log_likelihood) / y.shape[0]
	return loss

def feed_forward(mlp, x):

	# z = x * W(eights) + b(ias) with x == inputs
	# layer result a = activation(z)
	for i in range(len(mlp)):

		if (i == 0):
			#first layer: data input
			data = x
		else:
			data = mlp[i-1].a

		mlp[i].z = np.dot(mlp[i].W, data) + mlp[i].bias
		mlp[i].set_activation()

	return mlp[-1].softmax()

def back_propagation(mlp, loss):

	for i in range(len(mlp) - 1, -1, -1):
		if (i == len(mlp) - 1):
			error_prec = loss * mlp[-1].softmax_grad(mlp[-1].softmax())
			print(error_prec)
		else:
			error_prec = mlp[i+1].d

		mlp[i].d = np.dot(mlp[i].derivative_of_activation, error_prec)


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
