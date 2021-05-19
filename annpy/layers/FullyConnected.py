import annpy
from annpy.layers.Layer import Layer

import numpy as np

class FullyConnected(Layer):

	weights: np.ndarray		# -> kernel_shape
	bias: np.ndarray		# -> bias_shape

	inputs: np.ndarray		# Layer's input
	ws: np.ndarray			# Weighted sum result
	activation: np.ndarray	# Activation function result

	def __init__(self,
					output_shape,
					input_shape=None,
					activation='Linear',
					kernel_initializer='GlorotUniform',
					bias_initializer='Zeros',
					name="Default FCLayer name"):

		super().__init__(output_shape, input_shape, activation, kernel_initializer, bias_initializer, name)

	def compile(self, input_shape):

		self.kernel_shape = (input_shape, self.output_shape)

		self.weights = self.kernel_initializer(
			self.kernel_shape,
			input_shape=input_shape,
			output_shape=self.output_shape
		)
		self.bias = self.bias_initializer(
			self.bias_shape,
			input_shape=input_shape,
			output_shape=self.output_shape
		)

		return [self.weights, self.bias]

	def forward(self, inputs):

		self.inputs = inputs
		self.ws = np.dot(self.inputs, self.weights) + self.bias
		self.activation = self.fa(self.ws)

		return self.activation

	def backward(self, loss):
		"""
			3 partial derivatives
		"""
		# d(error) / d(activation)
		de = self.fa.derivate(self.ws)

		# d(error) / d(weighted sum)
		dfa = de * loss

		# d(error) / d(wi)
		dw = np.matmul(self.inputs.T, dfa) / self.inputs.shape[0]

		# d(error) / d(bias)
		db = np.mean(dfa, axis=0)

		# d(error) / d(xi)
		dx = np.matmul(dfa, self.weights.T) # (batch_size, n_neurons) * (n_neurons, n_input) = (batch_size, n_inputs)

		return dx, [dw, db]

		# print(f"inputs T {self.inputs.T.shape}:\n{self.inputs.T}")
		# print(f"weights {self.weights.shape}:\n{self.weights}")
		# print(f"ws {self.ws.shape}:\n{self.ws}")
		# print(f"activation {self.activation.shape}:\n{self.activation}")
		# print(f"loss {loss.shape}:\n{loss}")
		# print(f"de {de.shape}:\n{de}")
		# print(f"dfa      {dfa.shape}:\n{dfa}")
		# print(f"dfa T    {dfa.T.shape}:\n{dfa.T}")
		# print(f"dfa mean {dfa_mean.shape}:\n{dfa_mean}")
		# print(f"db {db.shape}:\n{db}")
		# print(f"dw {dw.shape}:\n{dw}")
		# print(f"dx {dx.shape}:\n{dx}")
		# exit(0)

	def _save(self):

		return {
			"type": "FullyConnected",
			"name": self.name,
			"activation": str(self.fa),
			"kernel": [list(w) for w in list(self.weights)],
			"bias": list(self.bias)
		}

	def summary(self):

		print(f"FCLayer {self.layer_index}: shape={self.weights.shape} + {self.bias.shape}")
		print(f"\tactivation = {self.fa},")
		print(f"\tkernel_initializer = {self.kernel_initializer},")
		print(f"\tbias_initializer = {self.bias_initializer}")
		print()
