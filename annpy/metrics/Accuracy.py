import numpy as np
from annpy.metrics.Metrics import Metrics

class Accuracy(Metrics):
	
	def __init__(self):
		super().__init__()
		pass

	def __call__(self, prediction, targets):
		# (batch_size, output_neuron)

		self.count += len([1 for predict, target in zip(prediction, targets) if np.array_equal(predict, target)])
		self.total += len(prediction)

	def get_result(self):
		return self.count / self.total

	def reset(self):
		accuracy = self.count / self.total
		self.count = 0
		self.total = 0
		return accuracy

	def summary(self):
		print(f"Accuracy:\t{self} (Exact values tested)")
