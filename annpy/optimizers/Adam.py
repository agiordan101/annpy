import numpy as np
from annpy.optimizers.Optimizer import Optimizer

class Adam(Optimizer):

	def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07):
		super().__init__(lr=lr)

		self.m1 = []
		self.m2 = []
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.beta_1_pow = 0
		self.beta_2_pow = 0
		self.epsilon = epsilon
		self.gradient_transform = self.adam

	def add(self, weightsB):
		self.m1.append([np.zeros(w.shape) for w in weightsB])
		self.m2.append([np.zeros(w.shape) for w in weightsB])

	def compile(self):
		# self.n_layers = len(self.m1)
		pass
	
	def adam(self, gradient, l, wi, **kwargs):

		# Moments
		self.m1[l][wi] = self.beta_1 * self.m1[l][wi] + (1 - self.beta_1) * gradient
		self.m2[l][wi] = self.beta_2 * self.m2[l][wi] + (1 - self.beta_2) * gradient * gradient

		# Correction
		m1_corrected = self.m1[l][wi] / (1 - self.beta_1_pow)
		m2_corrected = self.m2[l][wi] / (1 - self.beta_2_pow)
		self.beta_1_pow *= self.beta_1
		self.beta_2_pow *= self.beta_2

		return -self.lr * m1_corrected / (self.epsilon + np.sqrt(m2_corrected))

	def summary(self):
		print(f"Optimizer:\tannpy.optimizers.RMSProp, lr={self.lr}, momentum={self.beta_1}, rho={self.beta_2}, epsilon={self.epsilon}")
