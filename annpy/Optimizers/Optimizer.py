from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):

	def __init__(self, lr=0.1):

		self.obj_type = "Optimizer"
		self.lr = lr

	@abstractmethod
	def __call__(self):
		pass

	# @abstractmethod
	# def __str__(self):
	# 	pass
