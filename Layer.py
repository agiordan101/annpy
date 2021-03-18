import numpy as np
import activations

class Layer():

	def __init__(self,
					output_shape,
					input_shape=None,
					activation=activations.linear,
					name="Default layers name"):

		self.input_shape = input_shape
		self.output_shape = output_shape
		self.activation = activation
		self.name = name

	# def init_weights(self, input_shape):

	# 	if not self.input_shape:
	# 		self.input_shape = input_shape

	# 	self.weights = np.array(self.input_shape, self.output_shape)

	# 	return self.weights

	def compile(self, input_shape):
		# Create matrix weights
		# Link last layer output

		self.weights = np.random.rand(self.output_shape, input_shape)
		return self.weights

	def forward(self, inputs):

		return self.activation(np.dot(self.weights, inputs))

	# def backward(self):
	# 	pass

	def summary(self):
		
		print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		# print(f"weights {self.weights}")

