import numpy as np
from abc import ABCMeta, abstractmethod

class Activation(metaclass=ABCMeta):

	@abstractmethod
	def __call__(self, x):
		pass

	@abstractmethod
	def derivate(self, x):
		pass

	@abstractmethod
	def __str__(self):
		pass
