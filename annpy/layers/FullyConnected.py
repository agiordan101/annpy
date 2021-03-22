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
		self.summary()

		self.inputs = inputs
		self.ws = np.matmul(self.inputs, self.weights) + self.bias
		self.activation = self.fa(self.ws)
		# print(f"inputs {self.activation.shape}:\n{self.activation}")
		print(f"Inputs shape: {self.inputs.shape} : {self.inputs}")

		# self.weights = np.array([[1, 0, 0, 1], [0, 1, 1, 1]])
		# self.bias = np.array([10, 20, 30, 40])
		# print(f"inputs {self.inputs.shape}:\n{self.inputs}")
		# print(f"weights {self.weights.shape}:\n{self.weights}")
		# print(f"bias {self.bias.shape}:\n{self.bias}")
		# print(f"np.dot {np.matmul(self.inputs, self.weights).shape}:\n{np.matmul(self.inputs, self.weights)}")
		# print(f"np.dot {(np.matmul(self.inputs, self.weights) + self.bias).shape}:\n{np.matmul(self.inputs, self.weights) + self.bias}")

		print(f"Output shape: {self.activation.shape}: {self.activation}")
		return self.activation

	def backward(self, loss):
		"""
			3 partial derivatives
		"""
		self.summary()
		print(f"loss {loss.shape}:\n{loss}")

		print(f"inputs {self.inputs.shape}:\n{self.inputs}")
		print(f"weights {self.weights.shape}:\n{self.weights}")
		print(f"activation {self.activation.shape}:\n{self.activation}")

		# d(error) / d(activation)
		de = self.fa.derivate(self.activation)
		print(f"de = faderiv(activation) {de.shape}:\n{de}")

		# d(activation) / d(weighted sum)
		dfa = de * loss
		print(f"dfa = de * loss {dfa.shape}:\n{dfa}")

		# d(weighted sum) / d(wi)
		dw = self.inputs.T * dfa			#Tranpose necessary to update correct weights in neuron
		print(f"dw = inputs.T * dfa{dw.shape}:\n{dw}")

		# d(weighted sum) / d(bias)
		db = dfa
		# print(f"db {db.shape}:\n{db}")

		# d(weighted sum) / d(xi)
		dx = self.weights.T * dfa
		print(f"dx = weight * dfa {dx.shape}:\n{dx}")
		exit(0)
		return dx, dw, db

	def summary(self):
		
		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer: shape={self.weights.shape} + {self.bias.shape}, activation={self.fa}")
		# print(f"weights {self.weights}")
