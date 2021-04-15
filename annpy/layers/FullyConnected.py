import annpy
import numpy as np
from annpy.layers.Layer import Layer
from annpy.activations.Linear import Linear

class FullyConnected(Layer):

	def __init__(self,
					output_shape,
					input_shape=None,
					activation=Linear,
					kernel_initializer='GlorotUniform',
					bias_initializer='Zeros',
					name="Default layers name"):

		super().__init__(output_shape, input_shape, activation, kernel_initializer, bias_initializer, name)

	def compile(self, input_shape):
		# Link last layer output

		# self.weights = np.random.rand(input_shape + 1, self.output_shape)
		# self.weights = np.random.rand(input_shape, self.output_shape)
		# self.bias = np.random.rand(self.output_shape)

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
		# print(f"weights {self.weights.shape} / type {type(self.weights[0][0])} :\n{self.weights}")
		# print(f"bias {self.bias.shape} / type {type(self.bias[0])} :\n{self.bias}")
		# exit(0)
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
		# dfa_mean = np.mean(dfa, axis=0)
		# print(f"dfa      {dfa.shape}:\n{dfa}")
		# print(f"dfa T    {dfa.T.shape}:\n{dfa.T}")
		# print(f"dfa mean {dfa_mean.shape}:\n{dfa_mean}")

		# d(weighted sum) / d(wi)
		dw = np.matmul(self.inputs.T, dfa) / self.inputs.shape[0]		# (n_inputs, batch_size) * (batch_size, n_neurons) = (n_inputs, n_neurons)
		# dw = self.inputs.T * dfa
		# print(f"dw {dw.shape}:\n{dw}")

		# d(weighted sum) / d(bias)
		db = np.mean(dfa, axis=0)
		# print(f"db {db.shape}:\n{db}")

		# d(weighted sum) / d(xi)
		# dx = self.weights * np.mean(dfa, axis=0)
		dx = np.matmul(self.weights, dfa.T)	# (n_inputs, n_neurons) * (n_neurons, batch_size) = (n_inputs, batch_size?)
		# dx = np.mean(dfa, axis=0)
		# print(f"dx {dx.shape}:\n{dx}")

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
		return dx.T, [dw, db]

	def summary(self):

		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer {self.layer_index}: shape={self.weights.shape} + {self.bias.shape}")
		print(f"\tactivation = {self.fa},")
		print(f"\tkernel_initializer = {self.kernel_initializer},")
		print(f"\tbias_initializer = {self.bias_initializer}")
		print()
