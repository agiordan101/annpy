import numpy as np
from annpy.initializers.NormalInitializer import NormalInitializer

class GlorotNormal(NormalInitializer):

	def __call__(self, shape, input_shape, output_shape, **kwargs):

		print(f"kernel/bias init shape {shape}")
		self.stddev = np.sqrt(2 / (input_shape + output_shape))

		return super().__call__(shape)
