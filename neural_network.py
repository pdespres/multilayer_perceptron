import numpy as np

# Keras Arguments
# units: Positive integer, dimensionality of the output space.
# activation: Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
# use_bias: Boolean, whether the layer uses a bias vector.
# kernel_initializer: Initializer for the kernel weights matrix (see initializers).
# bias_initializer: Initializer for the bias vector (see initializers).
# kernel_regularizer: Regularizer function applied to the kernel weights matrix (see regularizer).
# bias_regularizer: Regularizer function applied to the bias vector (see regularizer).
# activity_regularizer: Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
# kernel_constraint: Constraint function applied to the kernel weights matrix (see constraints).
# bias_constraint: Constraint function applied to the bias vector (see constraints).

class Layer:
	
	batch_size = None

	def __init__(self, inputs, outputs, bias, layer_id):
	   
		self.layer_id   = layer_id
		self.inputs     = inputs
		self.outputs    = outputs
		self.bias       = bias
		self.size 		= inputs + bias

		# # summation vector
		# self.z = self.initialize_vector((self.inputs, Layer.batch_size))
				
		# activation vector
		## just inialize the vector, afterwards process as actual activation
		self.a = self.initialize_vector((self.inputs, Layer.batch_size))
		self.set_activation()        
		
		# weight matrix
		self.W = self.initialize_weights()        
		
		# # delta-error vector
		# self.d = self.initialize_vector((self.bias + self.inputs, Layer.batch_size))
		
		# # gradient error vector
		# self.g = self.initialize_vector(self.W.shape)

		# # gradient approximation vector
		# self.ga = self.initialize_vector(self.g.shape)

	def initialize_weights(self):
		# case there's none next layer as the output layer, also there's no Weight matrix
		if( self.outputs == None):
			return np.array([])
		weights = np.random.randn(self.outputs * self.size)
		weights = weights.reshape(self.outputs, self.size)
		return weights

	def initialize_vector(self, n_dimensions):
		return np.random.normal(size=n_dimensions)

	def set_activation(self):
		self.a = utils.fun_sigmoid(self.z)
		if self.bias: 
			self.add_activation_bias()


	
	# def add_activation_bias(self):
	# 	if len(self.a.shape) == 1:
	# 		self.a = np.vstack((1, self.a))
	# 	else:
	# 		self.a = np.vstack((np.ones(self.a.shape[1]), self.a))



	# def get_derivative_of_activation(self):
	# 	return utils.fun_sigmoid_derivative(self.a)



	# def update_weights(self, r):
	# 	self.W += -(r*self.g)



	# def check_gradient_computation(self, atol):
	# 	return np.allclose(self.g, self.ga, atol=atol)


		
	# def print_layer(self):
	# 	print "W:\n %s \n" %(self.W)
	# 	print "z: %s" %(self.z)
	# 	print "a: %s" %(self.a)
	# 	print "d: %s" %(self.d)
	# 	print "g: %s" %(self.g)