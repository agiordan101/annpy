import numpy as np
from annpy.initializers.NormalInitializer import NormalInitializer

class LecunNormal(NormalInitializer):

	def __call__(self, shape, input_shape):

		print(f"kernel/bias init shape {shape}")
		self.stddev = np.sqrt(1 / input_shape)

		return super().__call__(shape)
