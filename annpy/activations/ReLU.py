import numpy as np
from annpy.activations.Activation import Activation

class ReLU(Activation):

	def __call__(self, x):
		return np.where(x < 0, 0., x)

	def derivate(self, x):
		return np.where(x < 0, 0., 1.)

	def __str__(self):
		return "ReLU"
