import numpy as np
from annpy.losses.Loss import Loss

class BinaryCrossEntropy(Loss):

	def __init__(self, epsilon=1e-07):
		super().__init__()
		self.name = "BinaryCrossEntropy"
		self.epsilon = epsilon

	def BCE(self, prediction, target):
		""" BCE for one features/targets pair """

		# ret = np.mean(np.where(target == 1, np.log(prediction), np.log(1 - prediction)))
		ret = np.mean(target * np.log(prediction + self.epsilon) + (1 - target) * np.log(1 - prediction + self.epsilon))
		return ret

	def compute(self, predictions, targets):
		""" BCE for one batch of data"""
		bces = [self.BCE(prediction, target) for prediction, target in zip(predictions, targets)]
		return -np.mean(bces)

	def derivate(self, prediction, target):
		return prediction - target

	def summary(self, **kwargs):
		print(f"Metric:\tannpy.losses.BinaryCrossEntropy (Only for models with 2 output labels, 0 or 1)")
