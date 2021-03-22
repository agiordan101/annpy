import annpy
from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):

	obj_type = "Layer"

	def __init__(self,
					output_shape,
					input_shape=None,
					activation=annpy.activations.Linear(),
					name="Default layers name"):

		self.name = name
		self.input_shape = input_shape
		self.output_shape = output_shape
		# self.fa = annpy.utils.parse.parse_object(activation, annpy.activations.Activation, self.fas, annpy.activations.Linear)

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
