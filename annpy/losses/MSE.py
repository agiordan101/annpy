import numpy as np
from annpy.losses.Loss import Loss

class MSE(Loss):

	def __init__(self):
		pass

	def __call__(self, prediction, targets):
		# print(f"prediction {prediction.shape}:\n{prediction}")
		# print(f"targets {targets.shape}:\n{targets}")
		return np.mean(np.square(targets - prediction), axis=0)

	def summary(self):
		print(f"Loss:\t\t{self}")

	# def __str__(self):
	# 	return "MSE"