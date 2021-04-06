import numpy as np
from annpy.activations.Activation import Activation

class Softmax(Activation):

	def __call__(self, x):
		exp = np.exp(x)
		return exp / np.sum(exp, axis=1, keepdims=True)
		# return np.ma.fix_invalid(data, fill_value=1e+20).data

	# def __call__(self, x):
	# 	print(f"x=\n{x}")
	# 	exp = np.exp(x)
	# 	print(f"exp=\n{exp}")
	# 	sum = np.sum(exp, axis=1, keepdims=True)
	# 	print(f"sum=\n{sum}")
	# 	ret = exp / sum
	# 	print(f"ret=\n{ret}")
	# 	exit(0)
	# 	return ret

	def derivate(self, x):
		s = self(x)
		return s * (1 - s)

	# def __str__(self):
	# 	return "Softmax"
