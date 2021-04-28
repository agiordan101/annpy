import numpy as np
from annpy.activations.Activation import Activation

class Softmax(Activation):

	def __call__(self, x):
		exp = np.exp(x)
		return exp / np.sum(exp, axis=1, keepdims=True)

	def derivate(self, x):
		s = self(x)
		return s * (1 - s)

	def __str__(self):
		return "Softmax"
