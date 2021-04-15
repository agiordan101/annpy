import numpy as np
from annpy.activations.Activation import Activation

class Tanh(Activation):

	def __call__(self, x):
		return 1 - 2 / (np.exp(2 * x) + 1)

	def derivate(self, x):
		tanhx = self(x)
		return 1 - tanhx * tanhx
