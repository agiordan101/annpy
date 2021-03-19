import annpy
import numpy as np

fas = {
	"ReLU": annpy.activations.ReLU,
	"linear": annpy.activations.Linear,
	"Sigmoid": annpy.activations.Sigmoid,
}

class Layer():

	def __init__(self,
					output_shape,
					input_shape=None,
					activation=annpy.activations.Linear,
					name="Default layers name"):

		self.name = name
		self.input_shape = input_shape
		self.output_shape = output_shape

		self.fa = parse_object(activation, annpy.activations.Activation, fas, annpy.activations.Linear)

		# if hasattr(activation, "__call__") and hasattr(activation, "derivate"):
		# 	self.fa = activation()
		# else:
		# 	self.fa = fas.get(activation, annpy.activations.Linear)()
		# elif isinstance(activation, str):
		# 	self.activation = annpy.activations.Linear

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

		# d(activation) / d(weighted sum)	*	d(error) / d(activation)
		self.dfa = self.fa.derivate(self.ws) * loss
		print(f"self.dfa {self.dfa.shape}:\n{self.dfa}")

		# d(weighted sum) / d(wi)
		self.dw = self.inputs * self.dfa
		print(f"self.dw {self.dw.shape}:\n{self.dw}")

		# d(weighted sum) / d(xi)
		self.dx = self.weights * self.dfa
		print(f"self.dx {self.dx.shape}:\n{self.dx}")
		return self.dx

	def summary(self):
		
		# print(f"FCLayer: shape={self.weights.shape}, activation={self.activation}")
		print(f"FCLayer: shape={self.weights.shape} + {self.bias.shape}, activation={self.fa}")
		# print(f"weights {self.weights}")

