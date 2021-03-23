import numpy as np
from abc import ABCMeta, abstractmethod

class Loss(metaclass=ABCMeta):

	def __init__(self):
		pass

	@abstractmethod
	def __call__(self, prediction, targets):
		pass

	@abstractmethod
	def summary(self):
		pass

	# @abstractmethod
	# def __str__(self):
	# 	return "Loss"