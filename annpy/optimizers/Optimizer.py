from abc import ABCMeta, abstractmethod

class Optimizer(metaclass=ABCMeta):

	def __init__(self, lr=0.1):

		self.lr = lr
		self.gradients = []
		self.gradient_transform = None

	@abstractmethod
	def add(self, shape):
		pass

	def compile(self):

		print(f"ft : {self.gradient_transform}")
		if not self.gradient_transform:
			raise NotImplementedError

	def apply_gradients(self, weights):
		# weights:		[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[[dw, db], [..., ...], [wn, bn]]

		l = len(weights) - 1
		for weights_l, gradients_l in zip(weights[::-1], self.gradients):

			for i, gradient in enumerate(gradients_l):
				weights_l[i] += self.gradient_transform(gradient=gradient, l=l, wi=i)
			l -= 1

	@abstractmethod
	def summary(self):
		pass
