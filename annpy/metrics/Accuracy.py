import numpy as np
from annpy.metrics.Metrics import Metrics

class Accuracy(Metrics):
	
	def __init__(self):
		super().__init__()
		pass

	def __call__(self, prediction, targets, mask=None):
		
		mask = mask or np.full(len(prediction), 1)
		print(mask)

		valids = [1 for predict, target, desired in zip(prediction, targets, mask) if desired and np.array_equal(predict, target)]
		print(f"valids: {valids}")

		self.count += len(valids)
		self.total += len(prediction)
		self.accuracy = self.count / self.total

	def get_result(self):
		return self.accuracy

	def reset(self):
		self.count = 0
		self.total = 0

	def summary(self):
		print(f"Accuracy:\t{self} (Exact values tested)")
