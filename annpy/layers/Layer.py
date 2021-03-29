import annpy
from abc import ABCMeta, abstractmethod
from annpy.utils.parse import parse_object
from annpy.activations.Activation import Activation
from annpy.activations.Linear import Linear
from annpy.initializers.Initializer import Initializer

class Layer(metaclass=ABCMeta):


	def __init__(self, output_shape, input_shape, activation, kernel_initializer, bias_initializer, name):

		self.name = name
		self.layer_index = None
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.bias_shape = (1, output_shape)
		self.fa = parse_object(activation, Activation)
		self.kernel_initializer = parse_object(kernel_initializer, Initializer)
		self.bias_initializer = parse_object(bias_initializer, Initializer)

	# @abstractmethod
	# def __str__(self):
	# 	pass

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
