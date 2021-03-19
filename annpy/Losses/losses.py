import numpy as np

class MSE():

	def __init__(self):
		pass

	def __call__(self, prediction, targets):
		return np.mean(np.square(targets - prediction), axis=0)
