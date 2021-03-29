import numpy as np
from annpy.metrics.Accuracy import Accuracy

class RangeAccuracy(Accuracy):

	def __init__(self, ranges):
		super().__init__()

		if isinstance(ranges, list):
			self.ranges = ranges
		else:
			raise Exception(f"[RangeAccuracy] Ranges parameter is not a list (type={type(ranges)})")

	def get_obj_name(self):
		return "RangeAccuracy"

	def accuracy_conditions(self, prediction, target):
		return all([not (p < t - r or t + r < p) for p, t, r in zip(np.nditer(prediction), np.nditer(target), self.ranges)])

	def summary(self):
		print(f"Metric:\tannpy.accuracies.RangeAccuracy (An error interval around targets is allowed)")