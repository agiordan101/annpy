import numpy as np
from annpy.initializers.UniformInitializer import UniformInitializer

class LecunUniform(UniformInitializer):

	def __call__(self, shape, input_shape):

		print(f"kernel/bias init shape {shape}")
		self.max = np.sqrt(3 / input_shape)
		self.min = -self.max

		return super().__call__(shape)
