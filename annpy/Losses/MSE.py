import numpy as np
from annpy.losses.Loss import Loss

class MSE():

	def __init__(self):
		pass

	def __call__(self, prediction, targets):
		return np.mean(np.square(targets - prediction), axis=0)

	# def __str__(self):
	# 	return "MSE"