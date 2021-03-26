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
		# print(f"Inputs shape: {inputs.shape}")
		# print(f"weights {self.weights.shape}:\n{self.weights}")

		self.inputs = inputs
		self.ws = np.dot(self.inputs, self.weights) + self.bias
		self.activation = self.fa(self.ws)

		# print(f"Output shape: {self.activation.shape}")
		return self.activation

	def backward(self, loss):
		"""
			3 partial derivatives
		"""
		# print(f"BACKPROPAGATION")

		# print(f"inputs T {self.inputs.T.shape}:\n{self.inputs.T}")
		# print(f"weights {self.weights.shape}:\n{self.weights}")
		# print(f"ws {self.ws.shape}:\n{self.ws}")
		# print(f"activation {self.activation.shape}:\n{self.activation}")

		# print(f"loss {loss.shape}:\n{loss}")
		# d(error) / d(activation)
		de = self.fa.derivate(self.ws)
		# print(f"de {de.shape}:\n{de}")

		# d(activation) / d(weighted sum)
		dfa = de * loss
		# dfa = np.matmul(de.T, loss)
		dfa_mean = np.mean(dfa, axis=0)
		# print(f"dfa      {dfa.shape}:\n{dfa}")
		# print(f"dfa T    {dfa.T.shape}:\n{dfa.T}")
		# print(f"dfa mean {dfa_mean.shape}:\n{dfa_mean}")

		# d(weighted sum) / d(wi)
		dw = np.matmul(self.inputs.T, dfa)		# (n_inputs, batch_size) * (batch_size, n_neurons) = (n_inputs, n_neurons)
		# dw = self.inputs.T * dfa
		# print(f"dw {dw.shape}:\n{dw}")

		# d(weighted sum) / d(bias)
		db = dfa_mean
		# print(f"db {db.shape}:\n{db}")

		# d(weighted sum) / d(xi)
		# dx = self.weights * np.mean(dfa, axis=0)
		dx = np.matmul(self.weights, dfa.T)	# (n_inputs, n_neurons) * (n_neurons, batch_size) = (n_inputs, batch_size?)
		# dx = np.mean(dfa, axis=0)
		# print(f"dx {dx.shape}:\n{dx}")
		return dx.T, dw, db

	def summary(self):

		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer {self.layer_index}: shape={self.weights.shape} + {self.bias.shape}, activation={self.fa}")
		# print(f"weights {self.weights}")
