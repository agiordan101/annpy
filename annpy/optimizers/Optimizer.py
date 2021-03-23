from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):

	def __init__(self, lr=0.1):

		self.lr = lr
		self.gradients = []

	@abstractmethod
	def __call__(self):
		pass

	@abstractmethod
	def apply_gradients(self, weights_lst):
		# weights_lst:	[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[(dx, dw, db), ...]
		pass

	@abstractmethod
	def summary(self):
		pass

	# @abstractmethod
	# def __str__(self):
	# 	pass
