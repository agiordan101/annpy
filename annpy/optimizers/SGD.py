import numpy as np
from annpy.optimizers.Optimizer import Optimizer

class SGD(Optimizer):

	def __init__(self, lr=0.1, momentum=0.0):
		super().__init__(lr=lr)

		self.v = []
		self.momentum = momentum

		if self.momentum:
			self.gradient_transform = self.sgd_momentum
		else:
			self.gradient_transform = self.sgd

	def add(self, weightsB):
		if self.momentum:
			self.v.append([np.zeros(w.shape) for w in weightsB])

	def compile(self):
		self.n_layers = len(self.v)

	def sgd(self, gradient, **kwargs):
		# print(f"gradient {gradient.shape}:\n{gradient}")
		return -self.lr * gradient

	def sgd_momentum(self, gradient, l, wi):
		# print(f"v[l][w] {self.v[l][wi].shape}:\n{self.v[l][wi]}")
		self.v[l][wi] = self.momentum * self.v[l][wi] + self.sgd(gradient)
		return self.v[l][wi]

	def summary(self):
		print(f"Optimizer:\tannpy.optimizers.SGD, lr={self.lr}, momentum={self.momentum}")