from abc import ABCMeta, abstractmethod

class Loss(metaclass=ABCMeta):

	def __init__(self):
		self.count = 0
		self.total = 0
		pass

	@abstractmethod
	def __call__(self, prediction, targets):
		pass

	@abstractmethod
	def get_result(self):
		pass

	@abstractmethod
	def reset(self):
		pass

	@abstractmethod
	def summary(self):
		pass

	# @abstractmethod
	# def __str__(self):
	# 	return "Loss"