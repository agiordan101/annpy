import annpy
from abc import ABCMeta, abstractmethod
from annpy.utils.parse import parse_object
from annpy.activations.Activation import Activation
from annpy.activations.Linear import Linear

class Layer(metaclass=ABCMeta):

	obj_type = "Layer"

	def __init__(self, output_shape, input_shape, activation, name):

		self.name = name
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.fa = parse_object(activation, Activation)

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
