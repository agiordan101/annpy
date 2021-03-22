import annpy
import numpy as np
from annpy.layers.Layer import Layer
from annpy.activations.Linear import Linear

class FullyConnected(Layer):

	def __init__(self,
					output_shape,
					input_shape=None,
					activation=Linear(),
					name="Default layers name"):

		super().__init__(output_shape, input_shape, activation, name)

	# def __str__(self):
	# 	return "FullyConnected"

	def compile(self, input_shape):
		# Link last layer output

		# self.weights = np.random.rand(input_shape + 1, self.output_shape)
		self.weights = np.random.rand(input_shape, self.output_shape) * 2 - 1
		self.bias = np.random.rand(self.output_shape) * 2 - 1
		return [self.weights, self.bias]

	def forward(self, inputs):

		# return self.activation(np.dot(self.weights, inputs))
		print(f"Inputs shape: {inputs.shape}")

		self.inputs = inputs
		self.ws = np.dot(self.inputs, self.weights) + self.bias
		self.activation = self.fa(self.ws)

		print(f"Output shape: {self.activation.shape}")
		return self.activation

	def backward(self, loss):
		"""
			3 partial derivatives
		"""
		print(f"loss {loss.shape}:\n{loss}")

		# d(error) / d(activation)
		de = self.fa.derivate(self.ws)
		print(f"de {de.shape}:\n{de}")

		# d(activation) / d(weighted sum)
		dfa = de * loss
		print(f"dfa {dfa.shape}:\n{dfa}")

		# d(weighted sum) / d(wi)
		dw = self.inputs * dfa
		print(f"dw {dw.shape}:\n{dw}")
		
		# d(weighted sum) / d(bias)
		db = dfa
		print(f"db {db.shape}:\n{db}")

		# d(weighted sum) / d(xi)
		dx = self.weights * dfa
		print(f"dx {dx.shape}:\n{dx}")
		return dx, dw, db

	def summary(self):
		
		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer: shape={self.weights.shape} + {self.bias.shape}, activation={self.fa}")
		# print(f"weights {self.weights}")

