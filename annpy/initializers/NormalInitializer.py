import numpy as np
from annpy.initializers.Initializer import Initializer

class NormalInitializer(Initializer):

	def __init__(self, mean=0, stddev=0.1):
		self.mean = mean
		self.stddev = stddev

	def __call__(self, shape, **kwargs):
		# print(f"kernel/bias init shape {shape}")
		return np.random.normal(
			loc=self.mean,
			scale=self.stddev,
			size=shape
		)

