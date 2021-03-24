import numpy as np
from annpy.losses.Loss import Loss

class MSE(Loss):

	def __init__(self):
		super().__init__()

	def get_obj_name(self):
		return "MSE"

	def compute(self, prediction, target):
		return np.mean(np.square(target - prediction), axis=0)

	def get_mem_len_append(self, predictions, targets):
		return 1
	
	def derivate(self, prediction, target):
		return prediction - target

	def summary(self):
		print(f"Loss:\t\tannpy.losses.MSE")


	# def __str__(self):
	# 	return "MSE"