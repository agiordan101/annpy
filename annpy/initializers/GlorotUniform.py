import numpy as np
from annpy.initializers.UniformInitializer import UniformInitializer

class GlorotUniform(UniformInitializer):

	def __call__(self, shape, input_shape, output_shape, **kwargs):

		print(f"kernel/bias init shape {shape}")
		self.max = np.sqrt(6 / (input_shape + output_shape))
		self.min = -self.max

		return super().__call__(shape)
