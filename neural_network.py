#!/usr/bin/env python 3.6
# waloo le encoding: utf-8 de malade

import numpy as np

class Layer:

	def __init__(self, inputs, outputs, bias, init, activation, layer_id):
	   
		self.inputs     = inputs
		self.outputs    = outputs
		self.bias       = bias
		self.size 		= inputs + bias
		self.init 		= init
		self.activation = activation
		self.layer_id   = layer_id

		# z = x * W(eights) + b(ias) with x == inputs
		# layer result a = activation(z)

		# init
		self.z 			= self.initialize_vector(self.inputs)
		self.a 			= self.initialize_vector(self.inputs)
		self.set_activation()        
		self.W 			= self.initialize_weights()        
		
		# # delta-error vector
		# self.d = self.initialize_vector((self.bias + self.inputs, Layer.batch_size))
		
		# # gradient error vector
		# self.g = self.initialize_vector(self.W.shape)

		# # gradient approximation vector
		# self.ga = self.initialize_vector(self.g.shape)

	def initialize_weights(self):
		if self.init == 'uniform':
			weights = np.random.rand(self.outputs * self.inputs)
		else: #normal
			weights = np.random.randn(self.outputs * self.inputs)
		weights = weights.reshape(self.outputs, self.inputs)
		return weights

	def initialize_vector(self, n_dimensions):
		return np.random.normal(size=n_dimensions)

	def set_activation(self):
		if self.activation == 'sigmoid':
			self.a = 1 / (1 + np.exp(-self.z))
		elif self.activation == 'relu':
			self.a = max(0, self.z)
		else: #tanh
			self.a = (np.exp(self.z) - np.exp(-self.z)) / (np.exp(self.z) + np.exp(-self.z))

	# def get_derivative_of_activation(self):
	# 	return utils.fun_sigmoid_derivative(self.a)
# sigmoid ∇xJ=y(1−y)∇yJ 
# tanh	∇xJ=(1+y)(1−y)∇yJ
# relu	∇xJ=[y≥0]∇yJ

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

def net_constructer(features, categories, array_layers_dim, array_init, array_activation):
	net = []

	for i in np.arange(len(array_layers_dim), dtype=int):

		# first layer connected to all features
		if (i == 0):
			l = Layer(features, array_layers_dim[i], bias=True,  \
				init=array_init, activation=array_activation, layer_id=i)
			l.print_layer()
			net.append(l)

		if i == len(array_layers_dim) - 1:
			# output layer
			l = Layer(array_layers_dim[i], categories, bias=True, \
			init=array_init, activation=array_activation, layer_id=i+1)
		else:
			# other hidden layers
			l = Layer(array_layers_dim[i], array_layers_dim[i+1], bias=True, \
			init=array_init, activation=array_activation, layer_id=i+1)
			
		l.print_layer()
		net.append(l)

	return net
