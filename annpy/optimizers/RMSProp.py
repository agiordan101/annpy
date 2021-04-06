import numpy as np
from annpy.optimizers.Optimizer import Optimizer

class RMSProp(Optimizer):

	def __init__(self, lr=0.001, rho=0.9, momentum=0.0, epsilon=1e-07):
		super().__init__(lr=lr)

		self.sum = []
		self.v = []
		self.momentum = momentum
		self.rho = rho
		self.epsilon = epsilon

		if self.momentum:
			self.gradient_transform = self.rmsprop_momentum
		else:
			self.gradient_transform = self.rmsprop

	def add(self, weightsB):
		self.sum.append([np.zeros(w.shape) for w in weightsB])
		if self.momentum:
			self.v.append([np.zeros(w.shape) for w in weightsB])

	def compile(self):
		self.n_layers = len(self.v)

	def rmsprop(self, gradient, l, wi):
		# print(f"sum[i] {self.sum[l][wi].shape}:\n{self.sum[l][wi]}")
		# print(f"gradient {gradient.shape}:\n{gradient}")
		self.sum[l][wi] = self.rho * self.sum[l][wi] + (1 - self.rho) * gradient * gradient
		return -self.lr / (self.epsilon + np.sqrt(self.sum[l][wi])) * gradient

	def rmsprop_momentum(self, gradient, l, wi):
		self.v[l][wi] = self.momentum * self.v[l][wi] + self.rmsprop(gradient, l, wi)
		return self.v[l][wi]

	def summary(self):
		print(f"Optimizer:\tannpy.optimizers.RMSProp, lr={self.lr}, momentum={self.momentum}, rho={self.rho}, epsilon={self.epsilon}")
