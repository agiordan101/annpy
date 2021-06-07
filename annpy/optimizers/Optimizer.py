from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):

	def __init__(self, lr=0.1):

		self.lr = lr
		self.gradients = []
		self.gradient_transform = None # Child optimizer main function

	@abstractmethod
	def add(self, shape):
		pass

	@abstractmethod
	def compile(self):
		pass

	def apply_gradients(self, weights):
		# weights:		[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[[dw, db], [..., ...], [wn, bn]]

		# Start at the end
		l = len(weights) - 1
		for weights_l, gradients_l in zip(weights[::-1], self.gradients):

			for i, gradient in enumerate(gradients_l):
				weights_l[i] += self.gradient_transform(gradient=gradient, l=l, wi=i)
			l -= 1

	@abstractmethod
	def summary(self):
		pass
