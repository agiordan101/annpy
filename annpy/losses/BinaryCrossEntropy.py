import numpy as np
from annpy.losses.Loss import Loss

class BinaryCrossEntropy(Loss):

	def __init__(self):
		super().__init__()
		self.name = "BinaryCrossEntropy"

	def BCE(self, prediction, target):
		ret = np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
		# print(f"p: {prediction}\tt: {target}\tBCE: {ret}")
		return ret

	def compute(self, predictions, targets):
		bces = [self.BCE(prediction, target) for prediction, target in zip(predictions, targets)]
		ret = -np.mean(bces)
		# print(f"{self.name}:\n{ret}")
		# print(f"{self.name}:\n{bces}\n{ret}")
		# if ret == np.nan:
		# 	print("WTFF NANANNANANANAN")
		# 	exit(0)
		return ret
	# def get_mem_len_append(self, **kwargs):
	# 	return 1

	def derivate(self, prediction, target):
		return prediction - target

	def summary(self, **kwargs):
		print(f"Metric:\tannpy.losses.BinaryCrossEntropy (Only for models with 2 output labels, 0 or 1)")
