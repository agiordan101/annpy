import numpy as np
from activations import Activations

class Layer():

	def __init__(self,
					output_shape,
					input_shape=None,
					activation="linear",
					name="Default layers name"):

		self.name = name
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.activation, self.activation_deriv = Activations.get(activation, Activations["ReLU"])

	# def init_weights(self, input_shape):

	# 	if not self.input_shape:
	# 		self.input_shape = input_shape

	# 	self.weights = np.array(self.input_shape, self.output_shape)

	# 	return self.weights

	def compile(self, input_shape):
		# Link last layer output

		# self.weights = np.random.rand(input_shape + 1, self.output_shape)
		self.weights = np.random.rand(input_shape, self.output_shape) * 2 - 1
		self.bias = np.random.rand(self.output_shape) * 2 - 1
		return [self.weights, self.bias]

	def forward(self, inputs):

		# return self.activation(np.dot(self.weights, inputs))
		print(f"Inputs shape; {inputs.shape}")

		self.inputs = inputs
		self.ws = np.dot(self.inputs, self.weights) + self.bias
		self.activation = self.activation(self.ws)

		print(f"Output shape: {self.activation.shape}")
		return self.activation

	def backward(self, loss):

		return self.inputs * self.activation_deriv(self.ws) * loss

	def summary(self):
		
		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer: shape={self.weights.shape} + {self.bias.shape}, activation={self.activation}")
		# print(f"weights {self.weights}")

