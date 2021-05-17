import numpy as np
from annpy.initializers.Initializer import Initializer

class ConstantInitializer(Initializer):

	def __init__(self, val):
		self.val = val

	def __call__(self, shape, **kwargs):
		return np.full(shape, fill_value=self.val)
