import annpy
from annpy.utils.parse import parse_object
from annpy.activations.Activation import Activation
from annpy.initializers.Initializer import Initializer

from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):


	def __init__(self, output_shape, input_shape, activation, kernel_initializer, bias_initializer, name):

		self.name = name
		self.layer_index = None
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.bias_shape = (output_shape,)
		self.fa = parse_object(activation, Activation)
		self.kernel_initializer = parse_object(kernel_initializer, Initializer)
		self.bias_initializer = parse_object(bias_initializer, Initializer)

	def set_layer_index(self, i):
		self.layer_index = i

	@abstractmethod
	def compile(self, input_shape):
		pass

	@abstractmethod
	def forward(self, inputs):
		pass

	@abstractmethod
	def backward(self, loss):
		pass

	@abstractmethod
	def summary(self):
		pass
