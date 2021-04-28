import numpy as np
from annpy.activations.Activation import Activation

class Sigmoid(Activation):

	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def derivate(self, x):
		s = self(x)
		return s * (1 - s)

	def __str__(self):
		return "Sigmoid"
