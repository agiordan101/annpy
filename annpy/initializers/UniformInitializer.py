import numpy as np
from annpy.initializers.Initializer import Initializer

class UniformInitializer(Initializer):

	def __init__(self, min_val=-0.05, max_val=0.05):
		self.min = min_val
		self.max = max_val

	def __call__(self, shape, **kwargs):
		# print(f"kernel/bias init shape {shape}")
		return np.random.uniform(
			low=self.min,
			high=self.max,
			size=shape
		)
