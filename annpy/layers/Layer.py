import annpy

from annpy.activations.Activation import Activation
from annpy.initializers.Initializer import Initializer

from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):

	name: str
	layer_index: int

	input_shape: int
	output_shape: int

	kernel_shape: tuple
	bias_shape: tuple

	kernel_initializer: Initializer = None
	bias_initializer: Initializer = None
	fa: Activation

	def __init__(self, output_shape, input_shape, activation, kernel_initializer, bias_initializer, name):

		self.name = name

		self.input_shape = input_shape
		self.output_shape = output_shape
		
		self.kernel_shape = None
		self.bias_shape = output_shape

		if kernel_initializer:
			self.kernel_initializer = annpy.utils.parse.parse_object(kernel_initializer, Initializer)
		if bias_initializer:
			self.bias_initializer = annpy.utils.parse.parse_object(bias_initializer, Initializer)

		# print(f"layer init: {(output_shape, input_shape, activation, kernel_initializer, bias_initializer, name)}")
		self.fa = annpy.utils.parse.parse_object(activation, Activation)

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
