from abc import ABCMeta, abstractmethod

class Metrics(metaclass=ABCMeta):
	
	def __init__(self):
		self.total = 0
		self.count = 0
		pass

	@abstractmethod
	def __call__(self):
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