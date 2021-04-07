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

	def apply_gradients(self, weightsB):
		# weightsB:		[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[[dw, db], [..., ...], [wn, bn]]

		l = len(weightsB) - 1
		for weightsB_l, gradients_l in zip(weightsB[::-1], self.gradients):

			for i, gradient in enumerate(gradients_l):
				weightsB_l[i] += self.gradient_transform(gradient=gradient, l=l, wi=i)
			l -= 1

	@abstractmethod
	def summary(self):
		pass
