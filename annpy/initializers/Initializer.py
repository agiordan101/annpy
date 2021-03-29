import numpy as np
from abc import ABCMeta, abstractmethod

class Initializer(metaclass=ABCMeta):

	@abstractmethod
	def __call__(self):
		pass

