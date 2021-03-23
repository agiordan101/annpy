import numpy as np
from annpy.losses.Loss import Loss

class MSE(Loss):

	def __init__(self):
		super().__init__()
		pass

	def __call__(self, prediction, targets):
		# print(f"prediction {prediction.shape}:\n{prediction}")
		# print(f"targets {targets.shape}:\n{targets}")
		self.count += np.mean(np.square(targets - prediction), axis=0)
		self.total += 1

	def get_result(self):
		return self.count / self.total

	def reset(self):
		accuracy = self.count / self.total
		self.count = 0
		self.total = 0
		return accuracy

	def summary(self):
		print(f"Loss:\t\t{self}")

	# def __str__(self):
	# 	return "MSE"