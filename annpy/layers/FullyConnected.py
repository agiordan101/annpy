from annpy.layers.Layer import Layer

import numpy as np

class FullyConnected(Layer):

	weights: np.ndarray = None
	kernel: np.ndarray = None
	bias: np.ndarray = None

	inputs: np.ndarray		# Layer's input
	ws: np.ndarray			# Weighted sum result
	activation: np.ndarray	# Activation function result

	def __init__(self,
					output_shape,
					input_shape=None,
					activation='Linear',
					kernel=None,
					kernel_initializer='GlorotUniform',
					bias=None,
					bias_initializer='Zeros',
					name="Default FCLayer name"):

		# print(f"fc init: {(output_shape, input_shape, activation, kernel_initializer, bias_initializer, name)}")
		super().__init__(
			output_shape,
			input_shape,
			activation,
			None if kernel else kernel_initializer,
			None if bias else bias_initializer,
			name
		)
		if kernel:
			# print(f"kernel: {kernel}")
			self.kernel = np.array(kernel)
		if bias:
			# print(f"bias: {bias}")
			self.bias = np.array(bias)

	def compile(self, input_shape):

		self.input_shape = input_shape
		self.kernel_shape = (input_shape, self.output_shape)

		if self.kernel is None:
			self.kernel = self.kernel_initializer(
				self.kernel_shape,
				input_shape=input_shape,
				output_shape=self.output_shape
			)

		if self.bias is None:
			self.bias = self.bias_initializer(
				self.bias_shape,
				input_shape=input_shape,
				output_shape=self.output_shape
			)

		# print(f"self.weights: {self.kernel, self.bias}")
		self.weights = [self.kernel, self.bias]
		return self.weights
		# return [self.kernel, self.bias]

	def forward(self, inputs):

		self.inputs = inputs
		self.ws = np.dot(self.inputs, self.kernel) + self.bias
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
		dx = np.matmul(dfa, self.kernel.T) # (batch_size, n_neurons) * (n_neurons, n_input) = (batch_size, n_inputs)

		return dx, [dw, db]

		# print(f"inputs T {self.inputs.T.shape}:\n{self.inputs.T}")
		# print(f"weights {self.kernel.shape}:\n{self.kernel}")
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

	"""
		Summary
	"""

	def summary(self):

		print(f"FCLayer {self.layer_index} - {self.name}: shape={self.kernel.shape} + {self.bias.shape}")
		print(f"\tactivation = {self.fa},")
		print(f"\tkernel_initializer = {self.kernel_initializer},")
		print(f"\tbias_initializer = {self.bias_initializer}")
		print()

	"""
		Model save
	"""

	def _save(self):

		return {
			'type': "FullyConnected",
			'name': self.name,
			'units': self.bias_shape,
			'activation': str(self.fa),
			'kernel': [list(w) for w in list(self.kernel)],
			'bias': list(self.bias)
		}


