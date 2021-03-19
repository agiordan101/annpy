import numpy as np

class SGD():

	def __init__(self, lr=0.1):
		self.lr = lr

	def forward(self, weights, deriv):
		return weights - self.lr * deriv
